# risk/risk_manager.py
import logging
from typing import Dict, List, Any, Optional
import time
import math
from datetime import datetime

from config.app_config import AppConfig
from database.models.user import RiskLevel
from exchange.abstract_exchange import TradingSignal as TradeSignal

logger = logging.getLogger(__name__)

class RiskManager:
    """
    Risk management system for controlling trade risk
    """
    def __init__(self, config: AppConfig, db):
        self.config = config
        self.db = db
        
        # Get risk parameters from config
        self.default_risk_level = self.config.get('risk.default_risk_level', 'medium')
        self.max_open_positions = self.config.get('risk.max_open_positions', 10)
        self.max_position_size_percent = self.config.get('risk.max_position_size_percent', 5.0)
        self.default_stop_loss_percent = self.config.get('risk.default_stop_loss_percent', 2.0)
        
        # Drawdown protection thresholds
        self.drawdown_stages = {
            1: self.config.get('risk.drawdown_protection.stage1_threshold', 5.0),
            2: self.config.get('risk.drawdown_protection.stage2_threshold', 10.0),
            3: self.config.get('risk.drawdown_protection.stage3_threshold', 15.0),
            4: self.config.get('risk.drawdown_protection.stage4_threshold', 20.0)
        }
        
        # Risk level constants
        self.RISK_LEVEL_LOW = 1
        self.RISK_LEVEL_MEDIUM = 2
        self.RISK_LEVEL_HIGH = 3
        self.RISK_LEVEL_EXTREME = 4
        
        # User risk state cache
        self.user_risk_state = {}
        
        # Position correlation data
        self.correlation_matrix = {}
        
        # Reference to notification manager (may be set later)
        self.notification_manager = None
        
        logger.info("Risk manager initialized")
    
    def assess_portfolio_risk(self):
        """
        Perform a comprehensive assessment of portfolio risk across all active users.
        
        Analyzes current positions, market conditions, and potential exposures
        to provide risk metrics and trigger protective measures when necessary.
        
        Returns:
            bool: True if assessment completed successfully
        """
        try:
            logger.info("Performing scheduled portfolio risk assessment")
            
            # Get users with active positions
            active_users = self._get_users_with_positions()
            
            if not active_users:
                logger.info("No active positions found for risk assessment")
                return True
                
            assessment_results = {}
            for user_id in active_users:
                try:
                    # Get user risk state
                    risk_state = self._get_user_risk_state(user_id)
                    
                    # Force update of risk state
                    self._update_user_risk_state(user_id)
                    
                    # Get user data
                    user_repo = self.db.get_repository('user')
                    user = user_repo.get_user_by_id(user_id)
                    
                    if not user:
                        logger.warning(f"User {user_id} not found during risk assessment")
                        continue
                    
                    # Perform portfolio analysis
                    metrics = self._analyze_portfolio(user_id, risk_state)
                    
                    # Determine risk actions
                    actions = self._determine_risk_actions(user_id, metrics, risk_state)
                    
                    # Implement risk actions if needed
                    if actions:
                        self._implement_risk_actions(user_id, actions, risk_state)
                    
                    # Record assessment
                    assessment_results[user_id] = {
                        'metrics': metrics,
                        'actions': actions,
                        'timestamp': time.time()
                    }
                    
                    logger.info(f"Portfolio risk assessment completed for user {user_id}")
                    
                except Exception as e:
                    logger.error(f"Error assessing portfolio risk for user {user_id}: {str(e)}")
            
            # Store assessment results for reporting
            self._record_assessment_results(assessment_results)
            
            logger.info(f"Portfolio risk assessment completed for {len(assessment_results)} users")
            return True
            
        except Exception as e:
            logger.error(f"Error performing portfolio risk assessment: {str(e)}")
            return False
        
    def validate_signal(self, user, signal: TradeSignal) -> Optional[TradeSignal]:
        """
        Validate a trade signal against risk parameters
        
        Args:
            user: User object
            signal: Trade signal
            
        Returns:
            TradeSignal: Validated signal or None if signal is rejected
        """
        try:
            # Get user risk state
            risk_state = self._get_user_risk_state(user.id)
            
            # Check if trading is allowed for this user
            if not self._is_trading_allowed(user, risk_state):
                logger.warning(f"Trading not allowed for user {user.id}")
                return None
                
            # Calculate position size
            position_size = self._calculate_position_size(user, signal, risk_state)
            
            if position_size <= 0:
                logger.warning(f"Position size too small for user {user.id}")
                return None
                
            # Set position size in signal
            signal.quantity = position_size
            
            # Validate stop loss and take profit
            signal = self._validate_risk_levels(user, signal, risk_state)
            
            # Check if signal passes portfolio risk check
            if not self._check_portfolio_risk(user, signal, risk_state):
                logger.warning(f"Signal rejected due to portfolio risk for user {user.id}")
                return None
                
            logger.info(f"Validated signal for user {user.id}, symbol: {signal.symbol}, side: {signal.side}, quantity: {signal.quantity}")
            return signal
            
        except Exception as e:
            logger.error(f"Error validating signal for user {user.id}: {str(e)}")
            return None
            
    def _get_user_risk_state(self, user_id: int) -> Dict[str, Any]:
        """
        Get or create risk state for a user
        
        Args:
            user_id: User ID
            
        Returns:
            Dict: User risk state
        """
        if user_id not in self.user_risk_state:
            # Initialize risk state
            self.user_risk_state[user_id] = {
                'drawdown_stage': 0,
                'max_equity': 0,
                'current_equity': 0,
                'open_positions': {},
                'position_correlation': {},
                'last_updated': 0
            }
            
        # Check if update is needed
        current_time = time.time()
        if current_time - self.user_risk_state[user_id]['last_updated'] > 60:
            self._update_user_risk_state(user_id)
            
        return self.user_risk_state[user_id]
        
    def _update_user_risk_state(self, user_id: int) -> None:
        """
        Update risk state for a user
        
        Args:
            user_id: User ID
        """
        try:
            # Get user
            user_repo = self.db.get_repository('user')
            user = user_repo.get_user_by_id(user_id)
            
            if not user:
                logger.warning(f"User {user_id} not found")
                return
                
            # Get open positions
            position_repo = self.db.get_repository('position')
            open_positions = position_repo.get_open_positions_by_user(user_id)
            
            # Update risk state
            risk_state = self.user_risk_state[user_id]
            
            # Update equity
            risk_state['current_equity'] = user.equity
            
            if user.equity > risk_state['max_equity']:
                risk_state['max_equity'] = user.equity
                
            # Calculate drawdown
            if risk_state['max_equity'] > 0:
                drawdown_percent = (risk_state['max_equity'] - risk_state['current_equity']) / risk_state['max_equity'] * 100
                
                # Determine drawdown stage
                if drawdown_percent >= self.drawdown_stages[4]:
                    risk_state['drawdown_stage'] = 4
                elif drawdown_percent >= self.drawdown_stages[3]:
                    risk_state['drawdown_stage'] = 3
                elif drawdown_percent >= self.drawdown_stages[2]:
                    risk_state['drawdown_stage'] = 2
                elif drawdown_percent >= self.drawdown_stages[1]:
                    risk_state['drawdown_stage'] = 1
                else:
                    risk_state['drawdown_stage'] = 0
                    
            # Update open positions
            risk_state['open_positions'] = {
                position.symbol: {
                    'id': position.id,
                    'side': position.side.value,
                    'entry_price': position.average_entry_price,
                    'quantity': position.current_quantity,
                    'strategy': position.strategy
                }
                for position in open_positions
            }
            
            # Update position correlation
            self._update_position_correlation(user_id, list(risk_state['open_positions'].keys()))
            
            # Update timestamp
            risk_state['last_updated'] = time.time()
            
        except Exception as e:
            logger.error(f"Error updating risk state for user {user_id}: {str(e)}")
            
    def _update_position_correlation(self, user_id: int, symbols: List[str]) -> None:
        """
        Update position correlation for a user
        
        Args:
            user_id: User ID
            symbols: List of symbols
        """
        try:
            if not symbols:
                return
                
            risk_state = self.user_risk_state[user_id]
            
            # Get correlation matrix (simplified version)
            # In a real implementation, this would use historical price data
            for i, symbol1 in enumerate(symbols):
                for j, symbol2 in enumerate(symbols):
                    if i >= j:
                        continue
                        
                    key = f"{symbol1}_{symbol2}"
                    
                    # Use cached correlation or set a default
                    if key in self.correlation_matrix:
                        correlation = self.correlation_matrix[key]
                    else:
                        # Default correlation between different assets (moderate correlation)
                        correlation = 0.3
                        self.correlation_matrix[key] = correlation
                        
                    # Store in risk state
                    if 'position_correlation' not in risk_state:
                        risk_state['position_correlation'] = {}
                        
                    risk_state['position_correlation'][key] = correlation
                    
        except Exception as e:
            logger.error(f"Error updating position correlation for user {user_id}: {str(e)}")
            
    def _is_trading_allowed(self, user, risk_state: Dict[str, Any]) -> bool:
        """
        Check if trading is allowed for a user
        
        Args:
            user: User object
            risk_state: User risk state
            
        Returns:
            bool: True if trading is allowed, False otherwise
        """
        # Check if user is paused
        if user.is_paused:
            return False
            
        # Check drawdown stage
        if risk_state['drawdown_stage'] == 4:
            # Stage 4: Complete trading halt
            return False
        elif risk_state['drawdown_stage'] == 3:
            # Stage 3: Allow closing positions only
            # This is checked later in signal validation
            pass
        elif risk_state['drawdown_stage'] == 2:
            # Stage 2: Pause new entries
            # Allow only if closing an existing position
            if 'side' in risk_state and risk_state['side'].lower() == 'sell':
                return True
            return False
            
        # Check maximum open positions
        if len(risk_state['open_positions']) >= self.max_open_positions:
            logger.warning(f"Maximum open positions reached for user {user.id}")
            return False
            
        return True
        
    def _calculate_position_size(self, user, signal: TradeSignal, risk_state: Dict[str, Any]) -> float:
        """
        Calculate appropriate position size
        
        Args:
            user: User object
            signal: Trade signal
            risk_state: User risk state
            
        Returns:
            float: Position size
        """
        try:
            # Get user balance
            if hasattr(user, 'balance') and user.balance > 0:
                balance = user.balance
            else:
                # Get balance from exchange
                quote_currency = signal.symbol.split('/')[1]
                balance_data = self.exchange.get_balance(quote_currency)
                balance = balance_data.get('free', 0)
                
            if balance <= 0:
                logger.warning(f"Insufficient balance for user {user.id}")
                return 0
                
            # Determine base risk percentage based on user's risk level
            if hasattr(user, 'risk_level'):
                risk_level = user.risk_level.value if hasattr(user.risk_level, 'value') else user.risk_level
            else:
                risk_level = self.default_risk_level
                
            if risk_level == 'low':
                risk_percent = 1.0
            elif risk_level == 'medium':
                risk_percent = 2.0
            elif risk_level == 'high':
                risk_percent = 3.0
            else:
                risk_percent = 2.0
                
            # Apply drawdown reduction
            if risk_state['drawdown_stage'] == 1:
                # Stage 1: Reduce position sizing by 25%
                risk_percent *= 0.75
            elif risk_state['drawdown_stage'] >= 2:
                # Stage 2+: Reduce position sizing by 50%
                risk_percent *= 0.5
                
            # Calculate maximum position size
            max_position_size = balance * (self.max_position_size_percent / 100)
            
            # Calculate position size based on risk per trade
            if signal.stop_loss and signal.price:
                # Risk-based position sizing using stop loss
                risk_amount = balance * (risk_percent / 100)
                price_diff_percent = abs(signal.price - signal.stop_loss) / signal.price * 100
                
                if price_diff_percent > 0:
                    position_size = risk_amount / price_diff_percent
                else:
                    position_size = max_position_size * 0.5  # Default to half of max if no stop loss
            else:
                # Default position size
                position_size = balance * (risk_percent / 100) / signal.price
                
            # Ensure position size doesn't exceed maximum
            position_size = min(position_size, max_position_size / signal.price)
            
            # Round position size to appropriate precision
            position_size = self._round_position_size(position_size, signal.symbol)
            
            return position_size
            
        except Exception as e:
            logger.error(f"Error calculating position size for user {user.id}: {str(e)}")
            return 0
            
    def _round_position_size(self, position_size: float, symbol: str) -> float:
        """
        Round position size to appropriate precision for the symbol
        
        Args:
            position_size: Position size
            symbol: Trading pair symbol
            
        Returns:
            float: Rounded position size
        """
        # This is a simplified version
        # In a real implementation, this would get precision from exchange info
        if 'BTC' in symbol:
            return round(position_size, 6)
        elif 'ETH' in symbol:
            return round(position_size, 5)
        else:
            return round(position_size, 2)
            
    def _validate_risk_levels(self, user, signal: TradeSignal, risk_state: Dict[str, Any]) -> TradeSignal:
        """
        Validate and adjust stop loss and take profit levels
        
        Args:
            user: User object
            signal: Trade signal
            risk_state: User risk state
            
        Returns:
            TradeSignal: Updated signal
        """
        try:
            if not signal.stop_loss:
                # Set default stop loss
                signal.stop_loss = self.calculate_stop_loss(
                    signal.price, 
                    signal.side, 
                    self.default_stop_loss_percent
                )
                
            if not signal.take_profit:
                # Set default take profit at 2x the stop loss distance
                stop_loss_percent = abs(signal.price - signal.stop_loss) / signal.price * 100
                signal.take_profit = self.calculate_take_profit(
                    signal.price,
                    signal.side,
                    stop_loss_percent * 2
                )
                
            return signal
            
        except Exception as e:
            logger.error(f"Error validating risk levels for user {user.id}: {str(e)}")
            return signal
            
    def calculate_stop_loss(self, price: float, side: str, percent: float) -> float:
        """
        Calculate stop loss price
        
        Args:
            price: Entry price
            side: Trade side ('buy' or 'sell')
            percent: Stop loss percentage
            
        Returns:
            float: Stop loss price
        """
        if side.lower() == 'buy':
            return price * (1 - percent / 100)
        else:
            return price * (1 + percent / 100)
            
    def calculate_take_profit(self, price: float, side: str, percent: float) -> float:
        """
        Calculate take profit price
        
        Args:
            price: Entry price
            side: Trade side ('buy' or 'sell')
            percent: Take profit percentage
            
        Returns:
            float: Take profit price
        """
        if side.lower() == 'buy':
            return price * (1 + percent / 100)
        else:
            return price * (1 - percent / 100)
            
    def _check_portfolio_risk(self, user, signal: TradeSignal, risk_state: Dict[str, Any]) -> bool:
        """
        Check if signal passes portfolio risk checks
        
        Args:
            user: User object
            signal: Trade signal
            risk_state: User risk state
            
        Returns:
            bool: True if signal passes, False otherwise
        """
        try:
            # Check if adding this position would exceed maximum allocation
            total_allocation = self._calculate_total_allocation(user, risk_state)
            
            # Calculate new allocation with this signal
            new_position_value = signal.price * signal.quantity
            new_allocation = (total_allocation + new_position_value) / risk_state['current_equity'] * 100
            
            # Check if allocation exceeds maximum
            max_allocation = 80  # 80% maximum allocation
            
            if new_allocation > max_allocation:
                logger.warning(f"Signal rejected: would exceed max allocation for user {user.id}")
                return False
                
            # Check correlation with existing positions
            if not self._check_correlation_risk(user, signal, risk_state):
                logger.warning(f"Signal rejected: high correlation risk for user {user.id}")
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Error checking portfolio risk for user {user.id}: {str(e)}")
            return False
            
    def _calculate_total_allocation(self, user, risk_state: Dict[str, Any]) -> float:
        """
        Calculate total allocation of open positions
        
        Args:
            user: User object
            risk_state: User risk state
            
        Returns:
            float: Total allocation value
        """
        total_value = 0
        
        for symbol, position in risk_state['open_positions'].items():
            position_value = position['entry_price'] * position['quantity']
            total_value += position_value
            
        return total_value
        
    def _check_correlation_risk(self, user, signal: TradeSignal, risk_state: Dict[str, Any]) -> bool:
        """
        Check correlation risk of new position with existing portfolio
        
        Args:
            user: User object
            signal: Trade signal
            risk_state: User risk state
            
        Returns:
            bool: True if correlation risk is acceptable, False otherwise
        """
        # Skip if no existing positions
        if not risk_state['open_positions']:
            return True
            
        # Skip if no correlation data
        if not risk_state['position_correlation']:
            return True
            
        # Check correlation with each existing position
        high_correlation_count = 0
        
        for existing_symbol in risk_state['open_positions'].keys():
            if existing_symbol == signal.symbol:
                # Same symbol, check if it's the opposite side (hedging)
                existing_side = risk_state['open_positions'][existing_symbol]['side']
                
                if existing_side != signal.side:
                    # Hedging, so it's okay
                    continue
                else:
                    # Adding to existing position, so check other correlations
                    pass
                    
            # Check correlation
            key = f"{signal.symbol}_{existing_symbol}"
            reverse_key = f"{existing_symbol}_{signal.symbol}"
            
            if key in risk_state['position_correlation']:
                correlation = risk_state['position_correlation'][key]
            elif reverse_key in risk_state['position_correlation']:
                correlation = risk_state['position_correlation'][reverse_key]
            else:
                # Default correlation
                correlation = 0.3
                
            # Count high correlations
            if correlation > 0.7:
                high_correlation_count += 1
                
        # Reject if too many high correlations
        if high_correlation_count > 2:
            return False
            
        return True
        
    def get_drawdown_stage(self, user_id: int) -> int:
        """
        Get current drawdown stage for a user
        
        Args:
            user_id: User ID
            
        Returns:
            int: Drawdown stage (0-4)
        """
        risk_state = self._get_user_risk_state(user_id)
        return risk_state['drawdown_stage']
        
    def get_max_position_size(self, user_id: int, symbol: str) -> float:
        """
        Get maximum position size for a symbol
        
        Args:
            user_id: User ID
            symbol: Trading pair symbol
            
        Returns:
            float: Maximum position size
        """
        try:
            user_repo = self.db.get_repository('user')
            user = user_repo.get_user_by_id(user_id)
            
            if not user:
                return 0
                
            risk_state = self._get_user_risk_state(user_id)
            
            # Determine base risk percentage based on user's risk level
            if hasattr(user, 'risk_level'):
                risk_level = user.risk_level.value if hasattr(user.risk_level, 'value') else user.risk_level
            else:
                risk_level = self.default_risk_level
                
            if risk_level == 'low':
                risk_percent = 1.0
            elif risk_level == 'medium':
                risk_percent = 2.0
            elif risk_level == 'high':
                risk_percent = 3.0
            else:
                risk_percent = 2.0
                
            # Apply drawdown reduction
            if risk_state['drawdown_stage'] == 1:
                # Stage 1: Reduce position sizing by 25%
                risk_percent *= 0.75
            elif risk_state['drawdown_stage'] >= 2:
                # Stage 2+: Reduce position sizing by 50%
                risk_percent *= 0.5
                
            # Calculate maximum position size
            balance = user.balance
            max_position_size = balance * (self.max_position_size_percent / 100)
            
            # Get symbol price
            # In a real implementation, this would get the price from the exchange
            price = 1000  # Placeholder
            
            # Convert to position size
            max_position_size /= price
            
            # Round position size
            max_position_size = self._round_position_size(max_position_size, symbol)
            
            return max_position_size
            
        except Exception as e:
            logger.error(f"Error getting max position size for user {user_id}: {str(e)}")
            return 0
    
    def _get_users_with_positions(self):
        """
        Get list of users with active positions.
        
        Returns:
            List[int]: User IDs with active positions
        """
        try:
            user_ids = set()
            
            # Get users from position repository
            position_repo = self.db.get_repository('position')
            positions = position_repo.get_all_active_positions()
            
            for position in positions:
                user_ids.add(position.user_id)
                
            return list(user_ids)
            
        except Exception as e:
            logger.error(f"Error getting users with positions: {str(e)}")
            return []
    
    def _analyze_portfolio(self, user_id, risk_state):
        """
        Analyze portfolio risk metrics for a user.
        
        Args:
            user_id: User ID
            risk_state: User's risk state
            
        Returns:
            dict: Portfolio risk metrics
        """
        try:
            metrics = {
                'position_count': len(risk_state['open_positions']),
                'total_exposure': 0,
                'exposure_percent': 0,
                'highest_concentration': 0,
                'concentration_symbol': '',
                'portfolio_correlation': 0,
                'potential_drawdown': 0,
                'value_at_risk': 0,
                'overnight_exposure': 0,
                'volatility_exposure': 0
            }
            
            # Skip if no positions
            if not risk_state['open_positions']:
                return metrics
                
            # Calculate exposure
            total_value = 0
            largest_position_value = 0
            largest_position_symbol = ''
            
            for symbol, position in risk_state['open_positions'].items():
                position_value = position['entry_price'] * position['quantity']
                total_value += position_value
                
                if position_value > largest_position_value:
                    largest_position_value = position_value
                    largest_position_symbol = symbol
            
            # Calculate metrics
            metrics['total_exposure'] = total_value
            
            if risk_state['current_equity'] > 0:
                metrics['exposure_percent'] = (total_value / risk_state['current_equity']) * 100
                
            if total_value > 0:
                metrics['highest_concentration'] = (largest_position_value / total_value) * 100
                metrics['concentration_symbol'] = largest_position_symbol
                
            # Calculate portfolio correlation (average)
            if 'position_correlation' in risk_state and risk_state['position_correlation']:
                correlations = list(risk_state['position_correlation'].values())
                metrics['portfolio_correlation'] = sum(correlations) / len(correlations)
                
            # Calculate potential drawdown (simplified)
            # In a real implementation, this would use historical volatility data
            metrics['potential_drawdown'] = metrics['exposure_percent'] * 0.1  # Assume 10% max move
            metrics['value_at_risk'] = metrics['total_exposure'] * 0.05  # Assume 5% VaR
            
            # Check for overnight positions
            # In a real implementation, this would check market hours
            metrics['overnight_exposure'] = total_value
            
            # Volatility exposure (simplified)
            # In a real implementation, this would use actual volatility data
            metrics['volatility_exposure'] = total_value * 0.2  # Assume 20% volatility exposure
            
            return metrics
            
        except Exception as e:
            logger.error(f"Error analyzing portfolio for user {user_id}: {str(e)}")
            return {}
    
    def _determine_risk_actions(self, user_id, metrics, risk_state):
        """
        Determine actions needed based on risk assessment.
        
        Args:
            user_id: User ID
            metrics: Portfolio risk metrics
            risk_state: User's risk state
            
        Returns:
            List[dict]: Actions to take
        """
        try:
            actions = []
            
            # Check high exposure
            if metrics['exposure_percent'] > 70:
                actions.append({
                    'type': 'reduce_exposure',
                    'reason': 'high_exposure',
                    'threshold': 70,
                    'current': metrics['exposure_percent'],
                    'severity': 'high'
                })
                
            # Check high concentration
            if metrics['highest_concentration'] > 40:
                actions.append({
                    'type': 'reduce_concentration',
                    'reason': 'high_concentration',
                    'symbol': metrics['concentration_symbol'],
                    'threshold': 40,
                    'current': metrics['highest_concentration'],
                    'severity': 'medium'
                })
                
            # Check high correlation
            if metrics['portfolio_correlation'] > 0.7:
                actions.append({
                    'type': 'diversify',
                    'reason': 'high_correlation',
                    'threshold': 0.7,
                    'current': metrics['portfolio_correlation'],
                    'severity': 'medium'
                })
                
            # Check high potential drawdown
            if metrics['potential_drawdown'] > 15:
                actions.append({
                    'type': 'reduce_exposure',
                    'reason': 'high_drawdown_risk',
                    'threshold': 15,
                    'current': metrics['potential_drawdown'],
                    'severity': 'high'
                })
                
            # Check overnight exposure
            if metrics['overnight_exposure'] > 0.5 * risk_state['current_equity']:
                actions.append({
                    'type': 'reduce_overnight',
                    'reason': 'high_overnight_exposure',
                    'threshold': 50,
                    'current': (metrics['overnight_exposure'] / risk_state['current_equity']) * 100,
                    'severity': 'medium'
                })
                
            return actions
            
        except Exception as e:
            logger.error(f"Error determining risk actions for user {user_id}: {str(e)}")
            return []
    
    def _implement_risk_actions(self, user_id, actions, risk_state):
        """
        Implement risk mitigation actions.
        
        Args:
            user_id: User ID
            actions: List of actions to take
            risk_state: User's risk state
        """
        try:
            # Get user
            user_repo = self.db.get_repository('user')
            user = user_repo.get_user_by_id(user_id)
            
            if not user:
                logger.warning(f"User {user_id} not found during risk mitigation")
                return
                
            high_severity_actions = [a for a in actions if a['severity'] == 'high']
            medium_severity_actions = [a for a in actions if a['severity'] == 'medium']
            
            # Implement high severity actions
            if high_severity_actions:
                # Update user risk limits
                if hasattr(user, 'max_position_size_percent'):
                    ## Reduce position size limit
                    user.max_position_size_percent *= 0.75
                    user_repo.update_user(user)
                    
                # Update risk state
                risk_state['risk_reduction_active'] = True
                risk_state['risk_reduction_reason'] = high_severity_actions[0]['reason']
                risk_state['risk_reduction_expiry'] = time.time() + 86400  # 24 hours
                    
            # Send notification for all actions
            if hasattr(self, 'notification_manager') and self.notification_manager:
                self._send_risk_notification(user_id, actions)
                    
            logger.info(f"Implemented {len(actions)} risk actions for user {user_id}")
                
        except Exception as e:
            logger.error(f"Error implementing risk actions for user {user_id}: {str(e)}")
    
    def _send_risk_notification(self, user_id, actions):
        """
        Send risk notification to user.
        
        Args:
            user_id: User ID
            actions: List of actions taken
        """
        try:
            if not hasattr(self, 'notification_manager') or not self.notification_manager:
                return
                
            # Skip if no actions
            if not actions:
                return
                
            # Format message
            message = "Portfolio Risk Assessment Alert\n\n"
            
            # Add action details
            for action in actions:
                if action['type'] == 'reduce_exposure':
                    message += f"- High Total Exposure: {action['current']:.1f}% (Threshold: {action['threshold']}%)\n"
                elif action['type'] == 'reduce_concentration':
                    message += f"- High Concentration in {action['symbol']}: {action['current']:.1f}% (Threshold: {action['threshold']}%)\n"
                elif action['type'] == 'diversify':
                    message += f"- High Portfolio Correlation: {action['current']:.2f} (Threshold: {action['threshold']})\n"
                elif action['type'] == 'reduce_overnight':
                    message += f"- High Overnight Exposure: {action['current']:.1f}% (Threshold: {action['threshold']}%)\n"
            
            # Add recommendations
            message += "\nRecommended Actions:\n"
            
            if any(a['type'] == 'reduce_exposure' for a in actions):
                message += "- Consider reducing overall position sizes\n"
            if any(a['type'] == 'reduce_concentration' for a in actions):
                symbol = next(a['symbol'] for a in actions if a['type'] == 'reduce_concentration')
                message += f"- Consider reducing position size in {symbol}\n"
            if any(a['type'] == 'diversify' for a in actions):
                message += "- Consider adding uncorrelated assets to your portfolio\n"
            if any(a['type'] == 'reduce_overnight' for a in actions):
                message += "- Consider closing some positions before market close\n"
                
            # Send notification
            self.notification_manager.send_notification(
                user_id=user_id,
                message=message,
                notification_type="risk_assessment"
            )
            
        except Exception as e:
            logger.error(f"Error sending risk notification to user {user_id}: {str(e)}")
    
    def _record_assessment_results(self, results):
        """
        Record assessment results for reporting.
        
        Args:
            results: Assessment results
        """
        try:
            # In a real implementation, this would store results in a database
            # For now, we'll just log a summary
            high_risk_users = sum(1 for user_id, data in results.items() 
                                 if any(a['severity'] == 'high' for a in data.get('actions', [])))
            
            logger.info(f"Risk assessment summary: {len(results)} users assessed, {high_risk_users} high risk users")
            
        except Exception as e:
            logger.error(f"Error recording assessment results: {str(e)}")