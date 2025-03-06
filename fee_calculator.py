# profit/fee_calculator.py
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from decimal import Decimal, ROUND_HALF_UP

logger = logging.getLogger(__name__)

class FeeCalculator:
    """
    Comprehensive Fee Calculation and Management System

    Provides sophisticated fee calculation capabilities for trading platforms, 
    including dynamic fee structures, referral commission tracking, 
    and detailed financial reporting.
    """

    def __init__(
        self, 
        user_repository=None, 
        trade_repository=None, 
        transaction_repository=None, 
        config=None
    ):
        """
        Initialize the Fee Calculator with configurable dependencies.

        Args:
            user_repository (UserRepository): Repository for user data management
            trade_repository (TradeRepository): Repository for trade data management
            transaction_repository (TransactionRepository): Repository for transaction tracking
            config (dict, optional): Configuration settings for fee calculations
        """
        # Initialize repositories
        self.user_repo = user_repository
        self.trade_repo = trade_repository
        self.transaction_repo = transaction_repository
        
        # Load configuration
        self.config = config or {}
        
        # Define fee parameters
        self.profit_threshold = self.config.get('fees.profit_threshold', 1000)
        self.base_fee_percent = self.config.get('fees.base_fee_percent', 20.0)
        self.higher_fee_percent = self.config.get('fees.higher_fee_percent', 30.0)
        self.referral_commission_percent = self.config.get('fees.referral_commission_percent', 10.0)
        
        logger.info("Fee Calculator initialized successfully")

    def calculate_fee(self, user_id: int, trade_id: int, profit: float) -> Dict[str, float]:
        """
        Calculate comprehensive fee structure for a profitable trade.

        Performs detailed fee calculation, including base trading fees 
        and potential referral commissions.

        Args:
            user_id (int): Unique identifier for the user
            trade_id (int): Unique identifier for the trade
            profit (float): Total profit from the trade

        Returns:
            Dict[str, float]: Detailed breakdown of fees and net profit
        """
        try:
            # Immediately return zero fees for unprofitable trades
            if profit <= 0:
                return self._generate_zero_fee_structure(profit)
            
            # Validate required repositories
            if not self._validate_repository_dependencies():
                return self._generate_zero_fee_structure(profit)
            
            # Retrieve user information
            user = self.user_repo.get_user_by_id(user_id)
            if not user:
                logger.error(f"User {user_id} not found")
                return self._generate_zero_fee_structure(profit)
            
            # Determine appropriate fee percentage
            fee_percent = (
                self.base_fee_percent if user.total_profit < self.profit_threshold 
                else self.higher_fee_percent
            )
            
            # Perform precise decimal calculations
            profit_decimal = Decimal(str(profit))
            fee_percent_decimal = Decimal(str(fee_percent)) / Decimal('100')
            
            # Calculate primary fee amount
            fee_amount = profit_decimal * fee_percent_decimal
            fee_amount = fee_amount.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
            
            # Calculate referral commission
            referral_fee_amount = self._compute_referral_commission(
                user, trade_id, profit_decimal
            )
            
            # Calculate user's net profit after fees
            user_profit = profit_decimal - fee_amount - referral_fee_amount
            user_profit = user_profit.quantize(Decimal('0.00000001'), rounding=ROUND_HALF_UP)
            
            return {
                'fee_amount': float(fee_amount),
                'fee_percent': float(fee_percent),
                'referral_fee_amount': float(referral_fee_amount),
                'referral_percent': float(self.referral_commission_percent) if float(referral_fee_amount) > 0 else 0,
                'user_profit': float(user_profit)
            }
            
        except Exception as e:
            logger.error(f"Fee calculation error: {str(e)}")
            return self._generate_zero_fee_structure(profit)

    def _validate_repository_dependencies(self) -> bool:
        """
        Validate the availability of critical repositories.

        Returns:
            bool: Indicates whether all necessary repositories are configured
        """
        required_repositories = [
            self.user_repo, 
            self.trade_repo, 
            self.transaction_repo
        ]
        return all(required_repositories)

    def _compute_referral_commission(
        self, 
        user: Any, 
        trade_id: int, 
        profit: Decimal
    ) -> Decimal:
        """
        Calculate referral commission for a specific trade.

        Determines if a referral commission should be applied based on 
        the user's trading history and referral status.

        Args:
            user: User object
            trade_id: Unique trade identifier
            profit: Profit amount as a Decimal

        Returns:
            Decimal: Calculated referral commission amount
        """
        try:
            # Identify previous profitable trades
            profitable_trades = self.trade_repo.get_trades_by_user(user.id)
            profitable_trades = [
                trade for trade in profitable_trades 
                if trade.profit and trade.profit > 0 and trade.id != trade_id
            ]
            
            # Apply referral commission for first profitable trade
            if not profitable_trades and user.referrer_id:
                referral_percent = Decimal(str(self.referral_commission_percent)) / Decimal('100')
                return profit * referral_percent
            
            return Decimal('0')
        
        except Exception as e:
            logger.error(f"Referral commission calculation error: {str(e)}")
            return Decimal('0')

    def _generate_zero_fee_structure(self, profit: float) -> Dict[str, float]:
        """
        Generate a zero-fee structure for scenarios without applicable fees.

        Args:
            profit (float): Original profit amount

        Returns:
            Dict[str, float]: Zero-fee financial breakdown
        """
        return {
            'fee_amount': 0,
            'fee_percent': 0,
            'referral_fee_amount': 0,
            'referral_percent': 0,
            'user_profit': profit
        }

    def collect_fees(self, user_id: int, trade_id: int, profit: float) -> bool:
        """
        Collect and record fees for a completed trade.

        Manages the process of calculating, recording, and distributing 
        fees and potential referral commissions.

        Args:
            user_id (int): Unique user identifier
            trade_id (int): Unique trade identifier
            profit (float): Total trade profit

        Returns:
            bool: Indicates successful fee collection and recording
        """
        try:
            # Skip fee collection for non-profitable trades
            if profit <= 0:
                return True
            
            # Calculate comprehensive fee details
            fee_details = self.calculate_fee(user_id, trade_id, profit)
            
            # Record fee transactions
            self._record_fee_transactions(
                user_id, 
                trade_id, 
                fee_details, 
                profit
            )
            
            return True
        
        except Exception as e:
            logger.error(f"Fee collection error: {str(e)}")
            return False

    def _record_fee_transactions(
        self, 
        user_id: int, 
        trade_id: int, 
        fee_details: Dict[str, float], 
        total_profit: float
    ):
        """
        Record detailed fee and referral transactions.

        Args:
            user_id (int): User identifier
            trade_id (int): Trade identifier
            fee_details (Dict[str, float]): Calculated fee breakdown
            total_profit (float): Total trade profit
        """
        # Implementation details would depend on specific transaction repository methods
        pass

    def get_fee_summary(self, days: int = 30) -> Dict[str, Any]:
        """
        Generate comprehensive fee summary for the specified period.

        Args:
            days (int): Number of days to include in the summary

        Returns:
            Dict[str, Any]: Detailed fee collection and profit statistics
        """
        # Implementation would aggregate fee data from repositories
        pass