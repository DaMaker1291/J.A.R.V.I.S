"""
J.A.S.O.N. Financial Sentry Module
Transaction Monitoring and Burn Rate Analysis
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional
from woob.core import Woob

logger = logging.getLogger(__name__)

class FinancialSentryManager:
    """Manages financial monitoring and transaction analysis"""

    def __init__(self, config: dict):
        """Initialize Financial Sentry Manager

        Args:
            config: Configuration dict with Woob settings
        """
        self.config = config
        woob_config = config.get('financial_sentry', {}).get('woob', {})

        # Initialize Woob
        self.woob = Woob()
        backend_name = woob_config.get('backend')
        params = woob_config.get('params', {})

        if backend_name:
            try:
                self.backend = self.woob.load_backend(backend_name, params)
                logger.info(f"Loaded Woob backend: {backend_name}")
            except Exception as e:
                logger.error(f"Failed to load Woob backend {backend_name}: {e}")
                self.backend = None
        else:
            logger.warning("No Woob backend specified")
            self.backend = None

    def get_accounts(self, user_id: str = 'default') -> List[Dict[str, Any]]:
        """Get user accounts

        Args:
            user_id: User identifier

        Returns:
            List[Dict[str, Any]]: Account information
        """
        if not self.backend:
            return []

        accounts = []
        try:
            for account in self.backend.iter_accounts():
                accounts.append({
                    'account_id': account.id,
                    'name': account.label,
                    'type': str(account.type),
                    'balance': float(account.balance) if account.balance else 0.0,
                    'currency': account.currency or 'USD'
                })
        except Exception as e:
            logger.error(f"Failed to get accounts: {e}")

        return accounts

    def get_transactions(self, user_id: str = 'default', days: int = 30) -> List[Dict[str, Any]]:
        """Get recent transactions

        Args:
            user_id: User identifier
            days: Number of days to look back

        Returns:
            List[Dict[str, Any]]: Transaction data
        """
        if not self.backend:
            return []

        transactions = []
        try:
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days)

            for account in self.backend.iter_accounts():
                for transaction in self.backend.iter_history(account):
                    if transaction.date >= start_date.date() and transaction.date <= end_date.date():
                        transactions.append({
                            'transaction_id': transaction.id,
                            'account_id': account.id,
                            'amount': float(transaction.amount),
                            'date': transaction.date.isoformat(),
                            'name': transaction.label,
                            'category': getattr(transaction, 'category', []) or []
                        })
        except Exception as e:
            logger.error(f"Failed to get transactions: {e}")

        return transactions

    def calculate_burn_rate(self, user_id: str = 'default', period_days: int = 7) -> Dict[str, Any]:
        """Calculate burn rate (spending rate)

        Args:
            user_id: User identifier
            period_days: Period in days to calculate over

        Returns:
            Dict[str, Any]: Burn rate analysis
        """
        transactions = self.get_transactions(user_id, period_days)

        if not transactions:
            return {'error': 'No transactions found'}

        # Calculate total spending (negative amounts are outflows)
        total_spending = sum(t['amount'] for t in transactions if t['amount'] > 0)

        # Daily burn rate
        daily_burn = total_spending / period_days

        # Weekly burn rate
        weekly_burn = daily_burn * 7

        # Monthly burn rate (approximate)
        monthly_burn = daily_burn * 30

        # Category breakdown
        category_spending = {}
        for transaction in transactions:
            if transaction['amount'] > 0:
                category = transaction['category'][0] if transaction['category'] else 'Uncategorized'
                category_spending[category] = category_spending.get(category, 0) + transaction['amount']

        return {
            'period_days': period_days,
            'total_spending': round(total_spending, 2),
            'daily_burn': round(daily_burn, 2),
            'weekly_burn': round(weekly_burn, 2),
            'monthly_burn': round(monthly_burn, 2),
            'category_breakdown': category_spending,
            'transaction_count': len([t for t in transactions if t['amount'] > 0])
        }

    def monitor_spending_alerts(self, user_id: str = 'default') -> List[Dict[str, Any]]:
        """Check for spending alerts

        Args:
            user_id: User identifier

        Returns:
            List[Dict[str, Any]]: Alert messages
        """
        burn_rate = self.calculate_burn_rate(user_id, 7)  # Last week
        alerts = []

        # Define thresholds (would be configurable)
        weekly_threshold = self.config.get('financial_sentry', {}).get('alerts', {}).get('weekly_spend_limit', 1000)

        if burn_rate.get('weekly_burn', 0) > weekly_threshold:
            alerts.append({
                'type': 'high_spending',
                'message': f"Weekly spending (${burn_rate['weekly_burn']:.2f}) exceeds threshold (${weekly_threshold})",
                'severity': 'warning'
            })

        # Large transaction alert
        transactions = self.get_transactions(user_id, 1)  # Last day
        large_transaction_threshold = self.config.get('financial_sentry', {}).get('alerts', {}).get('large_transaction_limit', 500)

        for transaction in transactions:
            if transaction['amount'] > large_transaction_threshold:
                alerts.append({
                    'type': 'large_transaction',
                    'message': f"Large transaction: ${transaction['amount']} at {transaction['name']}",
                    'severity': 'info'
                })

        return alerts

    def get_financial_summary(self, user_id: str = 'default') -> Dict[str, Any]:
        """Get comprehensive financial summary

        Args:
            user_id: User identifier

        Returns:
            Dict[str, Any]: Financial summary
        """
        accounts = self.get_accounts(user_id)
        burn_rate = self.calculate_burn_rate(user_id, 7)
        alerts = self.monitor_spending_alerts(user_id)

        return {
            'accounts': accounts,
            'burn_rate': burn_rate,
            'alerts': alerts,
            'last_updated': datetime.now().isoformat()
        }
