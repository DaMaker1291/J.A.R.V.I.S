"""
J.A.S.O.N. Oracle Protocol
Predictive Simulation Engine: Monte Carlo Life Simulations
"""

import numpy as np
import random
from typing import Dict, Any, List, Callable, Optional
import time
from datetime import datetime, timedelta
import json
import os

class OracleManager:
    """Oracle Protocol: Monte Carlo predictive simulations for decision analysis"""

    def __init__(self, config: dict = None):
        self.config = config or {}

        # Oracle settings
        oracle_config = self.config.get('oracle', {})
        self.default_simulations = oracle_config.get('default_simulations', 10000)
        self.confidence_threshold = oracle_config.get('confidence_threshold', 0.8)
        self.risk_tolerance = oracle_config.get('risk_tolerance', 0.7)

        # Simulation results cache
        self.simulation_cache = {}
        self.max_cache_size = 100

        # Predefined simulation models
        self.simulation_models = {
            'financial': self._simulate_financial_decision,
            'meeting': self._simulate_meeting_outcome,
            'investment': self._simulate_investment,
            'career': self._simulate_career_move,
            'relationship': self._simulate_relationship_outcome,
            'health': self._simulate_health_decision,
            'travel': self._simulate_travel_decision,
            'purchase': self._simulate_purchase_decision,
            'negotiation': self._simulate_negotiation,
            'custom': self._simulate_custom_scenario
        }

        # Historical data for model calibration
        self.historical_data = self._load_historical_data()

    def _load_historical_data(self) -> Dict[str, Any]:
        """Load historical simulation data for model calibration"""
        data_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'oracle_historical.json')
        try:
            if os.path.exists(data_file):
                with open(data_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Failed to load historical data: {e}")
        return {}

    def _save_historical_data(self):
        """Save historical simulation data"""
        data_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'oracle_historical.json')
        os.makedirs(os.path.dirname(data_file), exist_ok=True)
        try:
            with open(data_file, 'w') as f:
                json.dump(self.historical_data, f, indent=2)
        except Exception as e:
            print(f"Failed to save historical data: {e}")

    def run_predictive_simulation(self, scenario: str, parameters: Dict[str, Any] = None,
                                num_simulations: int = None) -> Dict[str, Any]:
        """Run Monte Carlo predictive simulation for a decision scenario"""

        if num_simulations is None:
            num_simulations = self.default_simulations

        # Check cache first
        cache_key = f"{scenario}_{json.dumps(parameters, sort_keys=True)}"
        if cache_key in self.simulation_cache:
            return self.simulation_cache[cache_key]

        start_time = time.time()

        # Determine scenario type
        scenario_type = self._classify_scenario(scenario)

        if scenario_type not in self.simulation_models:
            return {
                'success': False,
                'error': f'Unknown scenario type: {scenario_type}',
                'supported_types': list(self.simulation_models.keys())
            }

        # Run simulations
        simulation_model = self.simulation_models[scenario_type]
        results = self._run_monte_carlo_simulation(simulation_model, parameters or {},
                                                 num_simulations, scenario)

        # Calculate statistics
        statistics = self._calculate_simulation_statistics(results)

        # Generate recommendations
        recommendations = self._generate_recommendations(statistics, scenario_type)

        result = {
            'success': True,
            'scenario': scenario,
            'scenario_type': scenario_type,
            'simulations_run': num_simulations,
            'execution_time': time.time() - start_time,
            'statistics': statistics,
            'recommendations': recommendations,
            'confidence_level': self._calculate_confidence_level(statistics),
            'risk_assessment': self._assess_risk(statistics),
            'timestamp': datetime.now().isoformat()
        }

        # Cache result
        if len(self.simulation_cache) >= self.max_cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.simulation_cache))
            del self.simulation_cache[oldest_key]

        self.simulation_cache[cache_key] = result

        # Update historical data
        self._update_historical_data(scenario_type, result)

        return result

    def _classify_scenario(self, scenario: str) -> str:
        """Classify scenario type based on keywords with typo resilience"""
        scenario_lower = scenario.lower()

        # Fuzzy keyword matching for typo resilience
        keywords = {
            'financial': ['stock', 'invest', 'buy', 'sell', 'market', 'financial', 'money', 'fianncial', 'monney', 'invst'],
            'meeting': ['meeting', 'presentation', 'interview', 'discussion', 'meetiing', 'interviiew', 'discusssion'],
            'investment': ['portfolio', 'diversify', 'allocation', 'returns', 'portfollo', 'diversiffy'],
            'career': ['job', 'career', 'promotion', 'switch', 'quit', 'carreer', 'promootion'],
            'relationship': ['relationship', 'marriage', 'breakup', 'date', 'relattonship', 'marrriage'],
            'health': ['health', 'medical', 'treatment', 'exercise', 'heallth', 'meddical'],
            'travel': ['travel', 'trip', 'vacation', 'flight', 'travvel', 'fligght', 'vacaation'],
            'purchase': ['buy', 'purchase', 'acquire', 'price', 'purchas'],
            'negotiation': ['negotiate', 'deal', 'contract', 'agreement', 'negotiiate', 'conttract']
        }

        for category, category_keywords in keywords.items():
            if any(word in scenario_lower for word in category_keywords):
                return category

        return 'custom'

    def _run_monte_carlo_simulation(self, model_func: Callable, parameters: Dict[str, Any],
                                  num_simulations: int, scenario: str) -> List[Dict[str, Any]]:
        """Run Monte Carlo simulations"""
        results = []

        for i in range(num_simulations):
            # Introduce randomness for Monte Carlo
            random.seed(i + int(time.time() * 1000) % 1000)

            try:
                # Run single simulation
                outcome = model_func(parameters.copy(), scenario)

                # Add random noise to simulate uncertainty
                outcome = self._add_uncertainty_noise(outcome)

                results.append(outcome)

            except Exception as e:
                print(f"Simulation {i} failed: {e}")
                continue

        return results

    def _add_uncertainty_noise(self, outcome: Dict[str, Any]) -> Dict[str, Any]:
        """Add realistic uncertainty to simulation outcomes"""
        # Add random noise to numerical outcomes
        for key, value in outcome.items():
            if isinstance(value, (int, float)) and key in ['probability', 'outcome', 'score']:
                # Add Gaussian noise (±10% standard deviation)
                noise = np.random.normal(0, abs(value) * 0.1)
                outcome[key] = max(0, min(1, value + noise))  # Clamp to [0,1] for probabilities

        return outcome

    def _calculate_simulation_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate comprehensive statistics from simulation results"""
        if not results:
            return {'error': 'No simulation results'}

        # Extract numerical outcomes
        outcomes = [r.get('outcome', r.get('probability', r.get('score', 0))) for r in results]

        # Basic statistics
        stats = {
            'mean': float(np.mean(outcomes)),
            'median': float(np.median(outcomes)),
            'std_dev': float(np.std(outcomes)),
            'min': float(np.min(outcomes)),
            'max': float(np.max(outcomes)),
            'total_simulations': len(results)
        }

        # Percentiles
        stats.update({
            'percentile_10': float(np.percentile(outcomes, 10)),
            'percentile_25': float(np.percentile(outcomes, 25)),
            'percentile_75': float(np.percentile(outcomes, 75)),
            'percentile_90': float(np.percentile(outcomes, 90))
        })

        # Probability distributions
        if outcomes:
            # Success rate (assuming outcomes > 0.5 are positive)
            success_rate = sum(1 for o in outcomes if o > 0.5) / len(outcomes)
            stats['success_probability'] = float(success_rate)

            # Risk assessment
            high_risk_count = sum(1 for o in outcomes if o < 0.3)
            stats['high_risk_probability'] = float(high_risk_count / len(outcomes))

            # Best/worst case scenarios
            best_case = max(outcomes)
            worst_case = min(outcomes)
            stats['best_case'] = float(best_case)
            stats['worst_case'] = float(worst_case)

        return stats

    def _generate_recommendations(self, statistics: Dict[str, Any], scenario_type: str) -> List[str]:
        """Generate actionable recommendations based on simulation results"""
        recommendations = []

        success_prob = statistics.get('success_probability', 0)
        risk_prob = statistics.get('high_risk_probability', 0)
        mean_outcome = statistics.get('mean', 0)

        # Success-based recommendations
        if success_prob > 0.8:
            recommendations.append("High probability of success - proceed with confidence")
        elif success_prob > 0.6:
            recommendations.append("Moderate probability of success - proceed with caution")
        elif success_prob > 0.4:
            recommendations.append("Low probability of success - consider alternatives")
        else:
            recommendations.append("High risk of failure - strongly recommend against proceeding")

        # Risk-based recommendations
        if risk_prob > 0.3:
            recommendations.append("High risk scenario detected - implement risk mitigation strategies")
        elif risk_prob > 0.1:
            recommendations.append("Moderate risk - monitor closely during execution")

        # Optimization recommendations
        std_dev = statistics.get('std_dev', 0)
        if std_dev > 0.2:
            recommendations.append("High outcome variability - consider diversification strategies")
        elif std_dev < 0.1:
            recommendations.append("Low outcome variability - results are predictable")

        # Scenario-specific recommendations
        if scenario_type == 'financial':
            if mean_outcome > 0.6:
                recommendations.append("Positive expected return - favorable investment opportunity")
            else:
                recommendations.append("Negative expected return - reconsider investment strategy")

        elif scenario_type == 'meeting':
            if success_prob > 0.7:
                recommendations.append("Meeting likely to go well - focus on key agenda items")
            else:
                recommendations.append("Prepare contingency plans for meeting challenges")

        return recommendations

    def _calculate_confidence_level(self, statistics: Dict[str, Any]) -> float:
        """Calculate overall confidence in simulation results"""
        # Based on sample size and statistical significance
        sample_size = statistics.get('total_simulations', 0)
        std_dev = statistics.get('std_dev', 1)

        # Confidence increases with sample size and decreases with variability
        base_confidence = min(1.0, sample_size / self.default_simulations)
        variability_penalty = min(0.5, std_dev)  # Cap penalty at 50%

        return base_confidence * (1 - variability_penalty)

    def _assess_risk(self, statistics: Dict[str, Any]) -> Dict[str, Any]:
        """Comprehensive risk assessment"""
        risk_prob = statistics.get('high_risk_probability', 0)
        std_dev = statistics.get('std_dev', 0)

        risk_level = 'low'
        if risk_prob > 0.3 or std_dev > 0.3:
            risk_level = 'high'
        elif risk_prob > 0.1 or std_dev > 0.15:
            risk_level = 'medium'

        risk_assessment = {
            'risk_level': risk_level,
            'risk_probability': float(risk_prob),
            'volatility': float(std_dev),
            'recommendations': []
        }

        if risk_level == 'high':
            risk_assessment['recommendations'].extend([
                "Implement comprehensive risk mitigation plan",
                "Consider insurance or hedging strategies",
                "Prepare contingency funding"
            ])
        elif risk_level == 'medium':
            risk_assessment['recommendations'].extend([
                "Monitor progress closely",
                "Have backup plans ready",
                "Consider phased implementation"
            ])

        return risk_assessment

    def _update_historical_data(self, scenario_type: str, result: Dict[str, Any]):
        """Update historical data for model calibration"""
        if scenario_type not in self.historical_data:
            self.historical_data[scenario_type] = []

        # Keep only recent results
        historical_results = self.historical_data[scenario_type]
        historical_results.append({
            'timestamp': result['timestamp'],
            'success_probability': result['statistics'].get('success_probability', 0),
            'mean_outcome': result['statistics'].get('mean', 0),
            'confidence': result.get('confidence_level', 0)
        })

        # Limit historical data
        if len(historical_results) > 100:
            historical_results.pop(0)

        self._save_historical_data()

    # ===== SIMULATION MODELS =====

    def _simulate_financial_decision(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate financial decision outcomes"""
        # Extract parameters
        amount = params.get('amount', 10000)
        timeframe = params.get('timeframe_months', 12)
        risk_level = params.get('risk_level', 'medium')

        # Base success factors
        market_trend = np.random.normal(0.05, 0.15)  # 5% mean return, 15% volatility
        personal_factor = np.random.normal(0.02, 0.1)  # Personal circumstances

        # Adjust for risk level
        if risk_level == 'high':
            market_trend *= 1.5  # Higher potential returns
            personal_factor *= 1.3  # Higher personal risk
        elif risk_level == 'low':
            market_trend *= 0.7  # Lower returns
            personal_factor *= 0.8  # Lower personal risk

        # Calculate outcome
        total_return = market_trend + personal_factor
        success_probability = max(0, min(1, 0.5 + total_return))  # Convert to probability

        return {
            'outcome': success_probability,
            'financial_return': total_return,
            'market_contribution': market_trend,
            'personal_contribution': personal_factor
        }

    def _simulate_meeting_outcome(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate meeting/interview outcomes"""
        preparation_level = params.get('preparation_hours', 5) / 10  # Normalize to 0-1
        stakeholder_count = min(params.get('stakeholders', 3), 10) / 10
        topic_complexity = params.get('complexity', 0.5)

        # Random factors
        chemistry_factor = np.random.beta(2, 2)  # Meeting chemistry
        timing_factor = np.random.normal(0.5, 0.2)  # Timing luck
        external_factors = np.random.normal(0, 0.1)  # Unexpected events

        # Calculate success
        base_success = (preparation_level * 0.4 + chemistry_factor * 0.3 +
                       timing_factor * 0.2 + (1 - topic_complexity) * 0.1)

        success_probability = max(0, min(1, base_success + external_factors))

        return {
            'outcome': success_probability,
            'preparation_impact': preparation_level * 0.4,
            'chemistry_factor': chemistry_factor,
            'timing_factor': timing_factor
        }

    def _simulate_investment(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate investment portfolio outcomes"""
        initial_amount = params.get('initial_amount', 100000)
        years = params.get('years', 10)
        allocation = params.get('allocation', {'stocks': 0.6, 'bonds': 0.3, 'cash': 0.1})

        # Simulate market returns
        stock_returns = np.random.normal(0.08, 0.15, years)  # 8% mean, 15% vol
        bond_returns = np.random.normal(0.03, 0.05, years)   # 3% mean, 5% vol
        cash_returns = np.random.normal(0.01, 0.005, years)  # 1% mean, 0.5% vol

        # Calculate portfolio returns
        portfolio_return = 0
        for year in range(years):
            yearly_return = (allocation.get('stocks', 0) * stock_returns[year] +
                           allocation.get('bonds', 0) * bond_returns[year] +
                           allocation.get('cash', 0) * cash_returns[year])
            portfolio_return += yearly_return

        # Convert to final amount
        final_amount = initial_amount * (1 + portfolio_return / years) ** years
        success_probability = 1 if final_amount > initial_amount * 1.5 else 0.5 if final_amount > initial_amount else 0

        return {
            'outcome': success_probability,
            'final_amount': final_amount,
            'total_return_percent': ((final_amount / initial_amount) - 1) * 100,
            'annualized_return': portfolio_return / years
        }

    def _simulate_career_move(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate career decision outcomes"""
        current_salary = params.get('current_salary', 75000)
        offered_salary = params.get('offered_salary', 85000)
        commute_time = params.get('commute_minutes', 30) / 60  # Convert to hours
        job_satisfaction = params.get('job_satisfaction', 0.7)

        # Financial calculation
        salary_increase = (offered_salary - current_salary) / current_salary

        # Quality of life factors
        commute_penalty = commute_time * 0.1  # 10% penalty per hour
        satisfaction_bonus = job_satisfaction * 0.2

        # Random factors
        career_growth = np.random.beta(2, 3)  # Potential for advancement
        work_life_balance = np.random.beta(2, 2)
        company_stability = np.random.beta(3, 2)

        # Overall success score
        success_score = (salary_increase * 0.3 + career_growth * 0.25 +
                        work_life_balance * 0.2 + satisfaction_bonus * 0.15 +
                        company_stability * 0.1 - commute_penalty)

        success_probability = max(0, min(1, success_score))

        return {
            'outcome': success_probability,
            'salary_impact': salary_increase,
            'career_growth_potential': career_growth,
            'work_life_balance': work_life_balance,
            'company_stability': company_stability
        }

    def _simulate_relationship_outcome(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate relationship decision outcomes"""
        compatibility_score = params.get('compatibility', 0.7)
        communication_level = params.get('communication', 0.6)
        shared_values = params.get('shared_values', 0.8)
        time_invested = min(params.get('time_invested_months', 6), 24) / 24  # Normalize

        # Random factors representing unpredictable relationship elements
        chemistry_spark = np.random.beta(2, 2)
        life_circumstances = np.random.normal(0, 0.1)
        external_pressures = np.random.normal(0, 0.15)

        # Relationship success model
        base_success = (compatibility_score * 0.25 + communication_level * 0.25 +
                       shared_values * 0.2 + chemistry_spark * 0.15 +
                       time_invested * 0.15)

        success_probability = max(0, min(1, base_success + life_circumstances + external_pressures))

        return {
            'outcome': success_probability,
            'compatibility_factor': compatibility_score,
            'communication_factor': communication_level,
            'chemistry_factor': chemistry_spark,
            'stability_factor': shared_values
        }

    def _simulate_health_decision(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate health-related decision outcomes"""
        treatment_type = params.get('treatment_type', 'conservative')
        severity_level = params.get('severity', 0.5)
        doctor_experience = params.get('doctor_experience', 0.8)

        # Base success rates by treatment type
        success_rates = {
            'conservative': 0.7,
            'moderate': 0.8,
            'aggressive': 0.85
        }

        base_success = success_rates.get(treatment_type, 0.7)

        # Adjust for severity and experience
        severity_penalty = severity_level * 0.3
        experience_bonus = doctor_experience * 0.2

        # Random factors
        patient_response = np.random.beta(2, 2)  # Individual response to treatment
        complication_chance = np.random.beta(1, 5)  # Low probability of complications

        success_probability = max(0, min(1, base_success - severity_penalty +
                                       experience_bonus + patient_response - complication_chance))

        return {
            'outcome': success_probability,
            'treatment_effectiveness': base_success,
            'severity_impact': -severity_penalty,
            'experience_bonus': experience_bonus,
            'patient_response': patient_response
        }

    def _simulate_travel_decision(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate travel decision outcomes"""
        destination_safety = params.get('destination_safety', 0.8)
        travel_cost = params.get('cost_level', 0.5)  # 0=cheap, 1=expensive
        trip_duration = min(params.get('duration_days', 7), 30) / 30
        purpose_value = params.get('purpose_value', 0.7)  # Personal/professional value

        # Random factors
        weather_luck = np.random.beta(3, 2)  # Good weather probability
        health_risks = np.random.beta(5, 1)  # Low health risk probability
        experience_quality = np.random.beta(2, 2)

        # Calculate overall success
        base_success = (destination_safety * 0.25 + (1 - travel_cost) * 0.2 +
                       trip_duration * 0.15 + purpose_value * 0.2 +
                       weather_luck * 0.1 + experience_quality * 0.1)

        # Health and safety penalties
        safety_penalty = (1 - destination_safety) * health_risks * 0.2

        success_probability = max(0, min(1, base_success - safety_penalty))

        return {
            'outcome': success_probability,
            'safety_factor': destination_safety,
            'experience_quality': experience_quality,
            'value_achievement': purpose_value,
            'weather_factor': weather_luck
        }

    def _simulate_purchase_decision(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate purchase decision outcomes"""
        item_quality = params.get('quality_rating', 0.8)
        price_reasonableness = params.get('price_fairness', 0.7)
        long_term_value = params.get('long_term_value', 0.6)
        urgency_level = params.get('urgency', 0.3)

        # Random factors
        market_timing = np.random.beta(2, 2)  # Good timing
        future_appreciation = np.random.beta(1.5, 2)  # Appreciation potential
        regret_probability = np.random.beta(1, 3)  # Low regret chance

        # Decision quality score
        decision_score = (item_quality * 0.25 + price_reasonableness * 0.25 +
                         long_term_value * 0.25 + market_timing * 0.15 +
                         future_appreciation * 0.1)

        # Urgency can lead to poor decisions
        urgency_penalty = urgency_level * 0.2

        success_probability = max(0, min(1, decision_score - urgency_penalty))

        return {
            'outcome': success_probability,
            'decision_quality': decision_score,
            'value_realization': long_term_value,
            'timing_factor': market_timing,
            'future_appreciation': future_appreciation
        }

    def _simulate_negotiation(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate negotiation outcomes"""
        your_position = params.get('your_position', 0.6)  # Negotiation leverage 0-1
        counterpart_toughness = params.get('counterpart_toughness', 0.5)
        time_pressure = params.get('time_pressure', 0.3)
        preparation_level = params.get('preparation', 0.7)

        # Random factors
        negotiation_skill = np.random.beta(2, 2)
        compromise_willingness = np.random.beta(2, 2)
        external_factors = np.random.normal(0, 0.1)

        # Negotiation success model
        leverage_advantage = your_position - counterpart_toughness
        skill_contribution = negotiation_skill * preparation_level

        base_success = (skill_contribution * 0.4 + leverage_advantage * 0.3 +
                       compromise_willingness * 0.2 + (1 - time_pressure) * 0.1)

        success_probability = max(0, min(1, base_success + external_factors))

        return {
            'outcome': success_probability,
            'negotiation_skill': negotiation_skill,
            'leverage_advantage': leverage_advantage,
            'compromise_factor': compromise_willingness,
            'preparation_bonus': preparation_level
        }

    def _simulate_custom_scenario(self, params: Dict[str, Any], scenario: str) -> Dict[str, Any]:
        """Simulate custom scenario using generic model"""
        # Extract any numerical parameters
        factors = []
        for key, value in params.items():
            if isinstance(value, (int, float)):
                factors.append(float(value))

        if not factors:
            # Default random simulation
            factors = [np.random.random() for _ in range(3)]

        # Simple weighted combination
        weights = [0.4, 0.3, 0.3]  # Adjust based on number of factors
        weights = weights[:len(factors)]

        outcome = sum(f * w for f, w in zip(factors, weights))

        # Add some randomness
        outcome += np.random.normal(0, 0.1)

        return {
            'outcome': max(0, min(1, outcome)),
            'factors_used': len(factors),
            'primary_factors': factors[:3]
        }

    def get_oracle_status(self) -> Dict[str, Any]:
        """Get Oracle Protocol status"""
        return {
            'supported_scenarios': list(self.simulation_models.keys()),
            'cache_size': len(self.simulation_cache),
            'historical_scenarios': len(self.historical_data),
            'default_simulations': self.default_simulations,
            'confidence_threshold': self.confidence_threshold
        }

    def clear_cache(self):
        """Clear simulation cache"""
        self.simulation_cache.clear()

    def get_scenario_history(self, scenario_type: str = None) -> Dict[str, Any]:
        """Get historical simulation data"""
        if scenario_type:
            return self.historical_data.get(scenario_type, [])
        else:
            return self.historical_data
