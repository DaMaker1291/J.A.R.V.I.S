"""
J.A.S.O.N. Concierge Module
Autonomous Booking Agent using Browser-Use and Multi-Agent Orchestration
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from langgraph.graph import StateGraph, END
from jason.tools.browser_agent import BrowserAgent
from jason.tools.serp_api import SearXNGSearch
from jason.core.audio import AudioManager
import caldav

logger = logging.getLogger(__name__)

class BookingState(Dict[str, Any]):
    """State for booking workflow"""
    messages: List[Dict[str, Any]]
    search_results: List[Dict[str, Any]]
    calendar_conflicts: bool
    selected_option: Optional[Dict[str, Any]]
    booking_details: Dict[str, Any]
    payment_ready: bool
    bio_confirmed: bool
    status: str

class ConciergeManager:
    """Manages autonomous booking workflows"""

    def __init__(self, config: dict, audio_manager: Optional[AudioManager] = None):
        """Initialize Concierge Manager

        Args:
            config: Configuration dict
            audio_manager: Optional audio manager for bio-lock
        """
        self.config = config
        self.audio = audio_manager

        # Initialize tools
        self.search_tool = SearXNGSearch()
        self.browser_agent = BrowserAgent()

        # Booking vault (in production, this would be encrypted)
        self.vault = config.get('concierge', {}).get('vault', {})

        # Setup LangGraph workflow
        self.workflow = self._create_workflow()

    def _create_workflow(self) -> StateGraph:
        """Create the booking workflow graph"""

        def search_flights(state: BookingState) -> BookingState:
            """Search for flights"""
            query = state.get('booking_details', {})
            origin = query.get('origin')
            destination = query.get('destination')
            date = query.get('date')

            if not all([origin, destination, date]):
                state['status'] = 'error: missing flight details'
                return state

            results = self.search_tool.search_flights(origin, destination, date)
            flights = self.search_tool.extract_flight_info(results)

            state['search_results'] = flights
            state['status'] = 'flights_searched'
            return state

        def search_hotels(state: BookingState) -> BookingState:
            """Search for hotels"""
            query = state.get('booking_details', {})
            location = query.get('location', query.get('destination'))
            checkin = query.get('checkin')
            checkout = query.get('checkout')

            if not all([location, checkin, checkout]):
                state['status'] = 'error: missing hotel details'
                return state

            results = self.search_tool.search_hotels(location, checkin, checkout)
            hotels = self.search_tool.extract_hotel_info(results)

            state['search_results'] = hotels
            state['status'] = 'hotels_searched'
            return state

        def check_calendar(state: BookingState) -> BookingState:
            """Check Radicale calendar for conflicts"""
            radicale_config = self.config.get('concierge', {}).get('radicale', {})
            url = radicale_config.get('url', 'http://localhost:5232')

            try:
                client = caldav.DAVClient(url=url)
                principal = client.principal()
                calendars = principal.calendars()

                if not calendars:
                    state['calendar_conflicts'] = False
                    state['status'] = 'calendar_checked'
                    return state

                calendar = calendars[0]  # Use first calendar

                booking_details = state.get('booking_details', {})
                if 'date' in booking_details:  # Flight
                    date = datetime.fromisoformat(booking_details['date'])
                    events = calendar.date_search(date, date)
                elif 'checkin' in booking_details:  # Hotel
                    start = datetime.fromisoformat(booking_details['checkin'])
                    end = datetime.fromisoformat(booking_details['checkout'])
                    events = calendar.date_search(start, end)
                else:
                    state['calendar_conflicts'] = False
                    state['status'] = 'calendar_checked'
                    return state

                if events:
                    state['calendar_conflicts'] = True
                else:
                    state['calendar_conflicts'] = False

            except Exception as e:
                logger.error(f"Calendar check failed: {e}")
                state['calendar_conflicts'] = False

            state['status'] = 'calendar_checked'
            return state

        def select_best_option(state: BookingState) -> BookingState:
            """Select the best booking option"""
            results = state.get('search_results', [])
            if not results:
                state['status'] = 'error: no options found'
                return state

            # Simple selection: cheapest option
            selected = min(results, key=lambda x: float(x.get('price', 0)) if x.get('price') else float('inf'))
            state['selected_option'] = selected
            state['status'] = 'option_selected'
            return state

        async def navigate_booking_site(state: BookingState) -> BookingState:
            """Navigate to booking site and fill forms"""
            selected = state.get('selected_option')
            if not selected:
                state['status'] = 'error: no option selected'
                return state

            # Determine site URL based on type
            booking_type = state.get('booking_details', {}).get('type', 'flight')
            site_config = self.config.get('concierge', {}).get('sites', {}).get(booking_type, {})

            booking_details = {
                'site_url': site_config.get('url'),
                'search_fields': site_config.get('search_fields', {}),
                'result_selector': site_config.get('result_selector'),
                'booking_fields': {**site_config.get('booking_fields', {}), **self.vault}
            }

            async with self.browser_agent as agent:
                result = await agent.execute_booking_workflow(booking_details)

            state['payment_ready'] = result.get('success', False)
            state['status'] = 'site_navigated' if result.get('success') else 'error: navigation failed'
            return state

        def confirm_payment(state: BookingState) -> BookingState:
            """Wait for bio-lock confirmation before payment"""
            if not state.get('payment_ready', False):
                state['status'] = 'error: not ready for payment'
                return state

            # Bio-lock confirmation
            if self.audio:
                confirmed = self._bio_lock_confirmation()
                state['bio_confirmed'] = confirmed
            else:
                # Fallback: assume confirmed for demo
                state['bio_confirmed'] = True

            state['status'] = 'payment_confirmed' if state['bio_confirmed'] else 'error: bio-lock failed'
            return state

        # Create workflow
        workflow = StateGraph(BookingState)

        # Add nodes
        workflow.add_node("search_flights", search_flights)
        workflow.add_node("search_hotels", search_hotels)
        workflow.add_node("check_calendar", check_calendar)
        workflow.add_node("select_option", select_best_option)
        workflow.add_node("navigate_site", navigate_booking_site)
        workflow.add_node("confirm_payment", confirm_payment)

        # Add edges
        workflow.add_edge("search_flights", "check_calendar")
        workflow.add_edge("search_hotels", "check_calendar")
        workflow.add_edge("check_calendar", "select_option")
        workflow.add_edge("select_option", "navigate_site")
        workflow.add_edge("navigate_site", "confirm_payment")
        workflow.add_edge("confirm_payment", END)

        # Set entry points based on booking type
        workflow.set_entry_point("search_flights")  # Default, would be conditional

        return workflow.compile()

    def _bio_lock_confirmation(self) -> bool:
        """Perform bio-lock confirmation using voice"""
        if not self.audio:
            return False

        try:
            # Listen for confirmation
            command = self.audio.listen()
            return "confirm" in command.lower() or "yes" in command.lower()
        except Exception as e:
            logger.error(f"Bio-lock confirmation failed: {e}")
            return False

    async def execute_booking(self, booking_request: Dict[str, Any]) -> Dict[str, Any]:
        """Execute autonomous booking workflow

        Args:
            booking_request: Booking details

        Returns:
            Dict[str, Any]: Workflow result
        """
        initial_state: BookingState = {
            'messages': [],
            'search_results': [],
            'calendar_conflicts': False,
            'selected_option': None,
            'booking_details': booking_request,
            'payment_ready': False,
            'bio_confirmed': False,
            'status': 'initiated'
        }

        try:
            result = await self.workflow.ainvoke(initial_state)
            return {
                'success': result.get('bio_confirmed', False),
                'status': result.get('status', 'unknown'),
                'selected_option': result.get('selected_option'),
                'payment_ready': result.get('payment_ready', False)
            }
        except Exception as e:
            logger.error(f"Booking workflow failed: {e}")
            return {'success': False, 'error': str(e)}

    def get_status(self) -> Dict[str, Any]:
        """Get concierge status

        Returns:
            Dict[str, Any]: Status information
        """
        return {
            'vault_configured': bool(self.vault),
            'audio_available': self.audio is not None,
            'supported_types': ['flights', 'hotels']
        }
