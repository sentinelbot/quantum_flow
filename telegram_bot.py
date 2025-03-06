# notification/telegram_bot.py
import logging
import threading
import time
import asyncio
import re
from typing import Dict, List, Any, Optional, Callable
import queue

import telebot
from telebot import types
from telebot.apihelper import ApiTelegramException

logger = logging.getLogger(__name__)

class TelegramBot:
    """
    Telegram bot for user interaction
    """
    def __init__(self, token: str, admin_chat_ids: List[int], api_key_repository=None, user_repository=None):
        self.token = token
        self.admin_chat_ids = admin_chat_ids or []
        self.bot = None
        self.running = False
        self.polling_thread = None
        self.api_key_repository = api_key_repository
        self.user_repository = user_repository
        
        # Command handlers
        self.command_handlers = {}
        
        # Message queue for rate limiting
        self.message_queue = queue.Queue()
        self.message_thread = None
        
        # State storage for conversation handling
        self.user_states = {}
        
        logger.info("Telegram bot initialized")
        
    def start(self) -> None:
        """
        Start Telegram bot
        """
        if not self.token:
            logger.warning("Telegram bot token not provided, bot will not start")
            return
            
        if self.running:
            logger.warning("Telegram bot already running")
            return
            
        try:
            logger.info("Starting Telegram bot")
            
            # Validate token format
            if not self._validate_token(self.token):
                logger.error("Invalid Telegram bot token format. Token must contain a colon.")
                return
                
            # Create bot instance
            self.bot = telebot.TeleBot(self.token)
            
            # Register command handlers
            self._register_handlers()
            
            # Start polling thread
            self.running = True
            self.polling_thread = threading.Thread(target=self._start_polling)
            self.polling_thread.daemon = True
            self.polling_thread.start()
            
            # Start message queue thread
            self.message_thread = threading.Thread(target=self._process_message_queue)
            self.message_thread.daemon = True
            self.message_thread.start()
            
            logger.info("Telegram bot started")
            
        except Exception as e:
            logger.error(f"Error starting Telegram bot: {str(e)}", exc_info=True)
            self.running = False
            
    def stop(self) -> None:
        """
        Stop Telegram bot
        """
        if not self.running:
            logger.warning("Telegram bot already stopped")
            return
            
        logger.info("Stopping Telegram bot")
        self.running = False
        
        if self.bot:
            try:
                self.bot.stop_polling()
            except Exception as e:
                logger.error(f"Error stopping Telegram bot polling: {str(e)}", exc_info=True)
                
        if self.polling_thread:
            self.polling_thread.join(timeout=5)
            
        if self.message_thread:
            self.message_thread.join(timeout=5)
            
        logger.info("Telegram bot stopped")
    
    def _validate_token(self, token: str) -> bool:
        """
        Validate Telegram bot token format
        
        Args:
            token: Telegram bot token
            
        Returns:
            bool: True if token format is valid, False otherwise
        """
        # Token must contain a colon (format: 123456789:ABCdefGHIjklMNoPQRstUVwxyz)
        if not token or not isinstance(token, str) or ":" not in token:
            return False
            
        # Basic validation - should contain bot number, colon, and alphanumeric API hash
        token_pattern = r'^\d+:[A-Za-z0-9_-]+$'
        return bool(re.match(token_pattern, token))
        
    def _start_polling(self) -> None:
        """
        Start bot polling with proper event loop handling
        """
        try:
            logger.info("Starting Telegram bot polling")
            
            # Set up new event loop for this thread if needed
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                # If no event loop exists in this thread, create one
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # Start polling in a way that works with asyncio
            self.bot.polling(non_stop=True, interval=1, timeout=20)
            
        except Exception as e:
            logger.error(f"Error in Telegram bot polling: {str(e)}", exc_info=True)
            self.running = False
            
    def _register_handlers(self) -> None:
        """
        Register command handlers
        """
        try:
            # Start command
            @self.bot.message_handler(commands=['start'])
            def handle_start(message):
                self._handle_start_command(message)
                
            # Help command
            @self.bot.message_handler(commands=['help'])
            def handle_help(message):
                self._handle_help_command(message)
                
            # Status command
            @self.bot.message_handler(commands=['status'])
            def handle_status(message):
                self._handle_status_command(message)
                
            # Register command
            @self.bot.message_handler(commands=['register'])
            def handle_register(message):
                self._handle_register_command(message)
                
            # Balance command
            @self.bot.message_handler(commands=['balance'])
            def handle_balance(message):
                self._handle_balance_command(message)
                
            # Pause command
            @self.bot.message_handler(commands=['pause'])
            def handle_pause(message):
                self._handle_pause_command(message)
                
            # Resume command
            @self.bot.message_handler(commands=['resume'])
            def handle_resume(message):
                self._handle_resume_command(message)
                
            # Risk command
            @self.bot.message_handler(commands=['risk'])
            def handle_risk(message):
                self._handle_risk_command(message)
                
            # Mode command
            @self.bot.message_handler(commands=['mode'])
            def handle_mode(message):
                self._handle_mode_command(message)
                
            # Performance command
            @self.bot.message_handler(commands=['performance'])
            def handle_performance(message):
                self._handle_performance_command(message)
                
            # Trades command
            @self.bot.message_handler(commands=['trades'])
            def handle_trades(message):
                self._handle_trades_command(message)
                
            # Open command
            @self.bot.message_handler(commands=['open'])
            def handle_open(message):
                self._handle_open_command(message)
                
            # Stats command
            @self.bot.message_handler(commands=['stats'])
            def handle_stats(message):
                self._handle_stats_command(message)
                
            # Report command
            @self.bot.message_handler(commands=['report'])
            def handle_report(message):
                self._handle_report_command(message)
                
            # Pairs command
            @self.bot.message_handler(commands=['pairs'])
            def handle_pairs(message):
                self._handle_pairs_command(message)
                
            # Strategies command
            @self.bot.message_handler(commands=['strategies'])
            def handle_strategies(message):
                self._handle_strategies_command(message)
                
            # Alerts command
            @self.bot.message_handler(commands=['alerts'])
            def handle_alerts(message):
                self._handle_alerts_command(message)
                
            # Admin commands (for admin users only)
            @self.bot.message_handler(commands=['admin'])
            def handle_admin(message):
                self._handle_admin_command(message)
                
            # Admin check command - NEW
            @self.bot.message_handler(commands=['admin_check'])
            def handle_admin_check(message):
                self._handle_admin_check_command(message)
                
            # Data command - NEW
            @self.bot.message_handler(commands=['data'])
            def handle_data(message):
                self._handle_data_command(message)
                
            # Setup exchange command - NEW
            @self.bot.message_handler(commands=['setup_exchange'])
            def handle_setup_exchange(message):
                self._handle_setup_exchange_command(message)
                
            # Text messages (for conversation flow)
            @self.bot.message_handler(content_types=['text'])
            def handle_text(message):
                self._handle_text_message(message)
                
            # Register callback query handler for inline buttons
            @self.bot.callback_query_handler(func=lambda call: True)
            def callback_query(call):
                self._handle_callback_query(call)
                
            logger.info("Telegram bot handlers registered")
            
        except Exception as e:
            logger.error(f"Error registering Telegram bot handlers: {str(e)}", exc_info=True)
    
    def _handle_callback_query(self, call):
        """
        Handle callback queries from inline keyboards
        
        Args:
            call: Callback query object
        """
        try:
            chat_id = call.message.chat.id
            data = call.data
            
            # Answer callback to remove loading indicator
            self.bot.answer_callback_query(call.id)
            
            # Process callback data
            if data.startswith("risk_"):
                risk_level = data.split("_")[1]
                self.send_message(chat_id, f"🛡️ Risk level has been set to *{risk_level.title()}*")
                
            elif data.startswith("mode_"):
                mode = data.split("_")[1]
                self.send_message(chat_id, f"🔄 Trading mode has been set to *{mode.title()}*")
                
            elif data.startswith("pair_"):
                action = data.split("_")[1]
                if action == "enable":
                    self.send_message(chat_id, "Please enter the symbol to enable:")
                    self.user_states[chat_id] = {"state": "awaiting_pair_enable", "data": {}}
                elif action == "disable":
                    self.send_message(chat_id, "Please enter the symbol to disable:")
                    self.user_states[chat_id] = {"state": "awaiting_pair_disable", "data": {}}
                elif action == "enable_all":
                    self.send_message(chat_id, "✅ All trading pairs have been enabled")
                elif action == "disable_all":
                    self.send_message(chat_id, "❌ All trading pairs have been disabled")
                    
            elif data.startswith("strategy_"):
                action = data.split("_")[1]
                if action == "enable":
                    self.send_message(chat_id, "Please select a strategy to enable:")
                    # TODO: Show strategy options
                elif action == "disable":
                    self.send_message(chat_id, "Please select a strategy to disable:")
                    # TODO: Show strategy options
                elif action == "adjust":
                    self.send_message(chat_id, "Please select a strategy to adjust:")
                    # TODO: Show strategy options
                    
            elif data.startswith("alert_"):
                alert_type = data.replace("alert_", "").replace("_", " ").title()
                self.send_message(chat_id, f"🔔 {alert_type} notifications have been toggled")
                
            elif data.startswith("setup_"):
                action = data.split("_")[1]
                if action == "exchange":
                    self.user_states[chat_id] = {"state": "awaiting_api_key", "data": {}}
                    self.send_message(chat_id, "Please enter your Binance API key:")
                
        except Exception as e:
            logger.error(f"Error handling callback query: {str(e)}", exc_info=True)
            
    def _handle_start_command(self, message) -> None:
        """
        Handle /start command
        """
        try:
            chat_id = message.chat.id
            
            welcome_message = (
                "👋 Welcome to *QuantumFlow Trading Bot*!\n\n"
                "I'm your cryptocurrency trading assistant. Here's what I can do for you:\n\n"
                "• Execute automated trading strategies\n"
                "• Track your portfolio performance\n"
                "• Send trading alerts and notifications\n"
                "• Manage your risk settings\n\n"
                "To get started, use /register to create an account.\n"
                "For a list of all commands, type /help."
            )
            
            self.send_message(chat_id, welcome_message)
            
        except Exception as e:
            logger.error(f"Error handling start command: {str(e)}", exc_info=True)
            
    def _handle_help_command(self, message) -> None:
        """
        Handle /help command
        """
        try:
            chat_id = message.chat.id
            
            help_message = (
                "📚 *QuantumFlow Trading Bot Commands*\n\n"
                "*Account Management:*\n"
                "/start - Start bot interaction\n"
                "/register - Begin registration process\n"
                "/verify - Complete verification steps\n"
                "/status - Show account status and settings\n"
                "/profile - Display user profile information\n"
                "/help - Show available commands\n\n"
                
                "*Trading Controls:*\n"
                "/balance - Show current exchange balance\n"
                "/pause - Temporarily halt all trading\n"
                "/resume - Restart trading activities\n"
                "/risk [low/medium/high] - Set risk level\n"
                "/mode [paper/live] - Switch trading mode\n\n"
                
                "*Performance Tracking:*\n"
                "/performance - Show overall trading results\n"
                "/trades - List recent trade history\n"
                "/open - Display current open positions\n"
                "/stats - Provide detailed performance metrics\n"
                "/report - Generate downloadable report\n"
                "/data - View current market data analysis\n\n"
                
                "*Settings Configuration:*\n"
                "/pairs - Manage tradable cryptocurrency pairs\n"
                "/strategies - Adjust strategy allocation\n"
                "/alerts - Configure notification preferences\n"
                "/setup_exchange - Configure your exchange API\n"
                "/admin_check - Verify admin status"
            )
            
            self.send_message(chat_id, help_message)
            
        except Exception as e:
            logger.error(f"Error handling help command: {str(e)}", exc_info=True)
            
    def _handle_status_command(self, message) -> None:
        """
        Handle /status command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual user status
            status_message = (
                "📊 *Bot Status*\n\n"
                "• Bot Status: *Running*\n"
                "• Trading Mode: *Paper Trading*\n"
                "• Risk Level: *Medium*\n"
                "• Active Strategies: *5*\n"
                "• Open Positions: *3*\n"
                "• Last Update: *5 minutes ago*\n\n"
                
                "💰 *Account Summary*\n\n"
                "• Balance: *$10,250.75*\n"
                "• Profit Today: *+$125.30 (1.23%)*\n"
                "• Total Profit: *+$1,250.75 (12.5%)*\n\n"
                
                "Use /performance for detailed metrics or /trades to see recent trades."
            )
            
            self.send_message(chat_id, status_message)
            
        except Exception as e:
            logger.error(f"Error handling status command: {str(e)}", exc_info=True)
            
    def _handle_register_command(self, message) -> None:
        """
        Handle /register command
        """
        try:
            chat_id = message.chat.id
            
            # Set user state for conversation flow
            self.user_states[chat_id] = {
                'state': 'awaiting_email',
                'data': {}
            }
            
            register_message = (
                "📝 *Registration Process*\n\n"
                "Let's get you set up with QuantumFlow Trading Bot.\n\n"
                "Please enter your email address:"
            )
            
            self.send_message(chat_id, register_message)
            
        except Exception as e:
            logger.error(f"Error handling register command: {str(e)}", exc_info=True)
            
    def _handle_balance_command(self, message) -> None:
        """
        Handle /balance command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual balance
            balance_message = (
                "💰 *Exchange Balance*\n\n"
                "*USDT:* $7,523.45\n"
                "*BTC:* 0.15 ($4,500.00)\n"
                "*ETH:* 1.2 ($3,000.00)\n"
                "*BNB:* 5.0 ($1,750.00)\n\n"
                "*Total Balance:* $16,773.45"
            )
            
            self.send_message(chat_id, balance_message)
            
        except Exception as e:
            logger.error(f"Error handling balance command: {str(e)}", exc_info=True)
            
    def _handle_pause_command(self, message) -> None:
        """
        Handle /pause command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Implement pause functionality
            pause_message = (
                "⏸️ *Trading Paused*\n\n"
                "All trading activities have been paused. The bot will not open any new positions, "
                "but will continue to monitor and manage existing positions.\n\n"
                "Use /resume to restart trading activities when you're ready."
            )
            
            self.send_message(chat_id, pause_message)
            
        except Exception as e:
            logger.error(f"Error handling pause command: {str(e)}", exc_info=True)
            
    def _handle_resume_command(self, message) -> None:
        """
        Handle /resume command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Implement resume functionality
            resume_message = (
                "▶️ *Trading Resumed*\n\n"
                "Trading activities have been resumed. The bot will now continue to "
                "monitor the market and execute trades according to your strategies.\n\n"
                "Use /status to check current bot status."
            )
            
            self.send_message(chat_id, resume_message)
            
        except Exception as e:
            logger.error(f"Error handling resume command: {str(e)}", exc_info=True)
            
    def _handle_risk_command(self, message) -> None:
        """
        Handle /risk command
        """
        try:
            chat_id = message.chat.id
            
            # Parse arguments
            args = message.text.split()
            
            if len(args) > 1:
                risk_level = args[1].lower()
                
                if risk_level in ['low', 'medium', 'high']:
                    # TODO: Update risk level
                    response = f"🛡️ Risk level has been set to *{risk_level.title()}*"
                else:
                    response = "❌ Invalid risk level. Use 'low', 'medium', or 'high'."
            else:
                # Show risk level options
                markup = types.InlineKeyboardMarkup()
                markup.row(
                    types.InlineKeyboardButton("Low Risk", callback_data="risk_low"),
                    types.InlineKeyboardButton("Medium Risk", callback_data="risk_medium"),
                    types.InlineKeyboardButton("High Risk", callback_data="risk_high")
                )
                
                response = (
                    "🛡️ *Risk Level Settings*\n\n"
                    "Select your preferred risk level:\n\n"
                    "• *Low Risk:* Conservative approach, smaller position sizes, tighter stop-losses\n"
                    "• *Medium Risk:* Balanced approach, moderate position sizes\n"
                    "• *High Risk:* Aggressive approach, larger position sizes, wider stop-losses\n\n"
                    "Current risk level: *Medium*"
                )
                
                self.send_message(chat_id, response, reply_markup=markup)
                return
                
            self.send_message(chat_id, response)
            
        except Exception as e:
            logger.error(f"Error handling risk command: {str(e)}", exc_info=True)
            
    def _handle_mode_command(self, message) -> None:
        """
        Handle /mode command
        """
        try:
            chat_id = message.chat.id
            
            # Parse arguments
            args = message.text.split()
            
            if len(args) > 1:
                mode = args[1].lower()
                
                if mode in ['paper', 'live']:
                    # TODO: Update trading mode
                    response = f"🔄 Trading mode has been set to *{mode.title()}*"
                else:
                    response = "❌ Invalid trading mode. Use 'paper' or 'live'."
            else:
                # Show mode options
                markup = types.InlineKeyboardMarkup()
                markup.row(
                    types.InlineKeyboardButton("Paper Trading", callback_data="mode_paper"),
                    types.InlineKeyboardButton("Live Trading", callback_data="mode_live")
                )
                
                response = (
                    "🔄 *Trading Mode Settings*\n\n"
                    "Select your preferred trading mode:\n\n"
                    "• *Paper Trading:* Simulated trading with virtual funds\n"
                    "• *Live Trading:* Real trading with actual funds\n\n"
                    "Current mode: *Paper Trading*"
                )
                
                self.send_message(chat_id, response, reply_markup=markup)
                return
                
            self.send_message(chat_id, response)
            
        except Exception as e:
            logger.error(f"Error handling mode command: {str(e)}", exc_info=True)
            
    def _handle_performance_command(self, message) -> None:
        """
        Handle /performance command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual performance data
            performance_message = (
                "📈 *Trading Performance*\n\n"
                "*Overall Statistics:*\n"
                "• Total Trades: *132*\n"
                "• Win Rate: *68.2%*\n"
                "• Profit Factor: *1.85*\n"
                "• Total Return: *+23.5%*\n"
                "• Max Drawdown: *5.7%*\n\n"
                
                "*Time-Based Performance:*\n"
                "• Today: *+1.2%*\n"
                "• This Week: *+3.7%*\n"
                "• This Month: *+8.5%*\n"
                "• This Year: *+23.5%*\n\n"
                
                "*Top Strategies:*\n"
                "• Trend Following: *+12.3%*\n"
                "• Mean Reversion: *+8.7%*\n"
                "• Grid Trading: *+5.8%*\n\n"
                
                "For detailed analysis, use /report to generate a complete performance report."
            )
            
            self.send_message(chat_id, performance_message)
            
        except Exception as e:
            logger.error(f"Error handling performance command: {str(e)}", exc_info=True)
            
    def _handle_trades_command(self, message) -> None:
        """
        Handle /trades command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual trade data
            trades_message = (
                "📊 *Recent Trades*\n\n"
                "*1. BTC/USDT* (Trend Following)\n"
                "Buy at $30,250 → Sell at $31,500\n"
                "Profit: +4.13% | 2 hours ago\n\n"
                
                "*2. ETH/USDT* (Mean Reversion)\n"
                "Buy at $1,850 → Sell at $1,920\n"
                "Profit: +3.78% | 5 hours ago\n\n"
                
                "*3. BNB/USDT* (Grid Trading)\n"
                "Sell at $355 → Buy at $345\n"
                "Profit: +2.90% | 8 hours ago\n\n"
                
                "*4. XRP/USDT* (Scalping)\n"
                "Buy at $0.52 → Sell at $0.50\n"
                "Loss: -3.85% | 12 hours ago\n\n"
                
                "*5. SOL/USDT* (Trend Following)\n"
                "Buy at $22.50 → Sell at $24.30\n"
                "Profit: +8.00% | 1 day ago\n\n"
                
                "Use /performance for overall statistics."
            )
            
            self.send_message(chat_id, trades_message)
            
        except Exception as e:
            logger.error(f"Error handling trades command: {str(e)}", exc_info=True)
            
    def _handle_open_command(self, message) -> None:
        """
        Handle /open command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual open positions
            open_positions_message = (
                "🔍 *Open Positions*\n\n"
                "*1. BTC/USDT* (Trend Following)\n"
                "Buy at $29,850 | Current: $30,200\n"
                "Unrealized P/L: +1.17%\n"
                "Stop Loss: $28,950 | Take Profit: $32,500\n\n"
                
                "*2. ETH/USDT* (Mean Reversion)\n"
                "Buy at $1,820 | Current: $1,845\n"
                "Unrealized P/L: +1.37%\n"
                "Stop Loss: $1,760 | Take Profit: $1,950\n\n"
                
                "*3. ADA/USDT* (Grid Trading)\n"
                "Buy at $0.42 | Current: $0.41\n"
                "Unrealized P/L: -2.38%\n"
                "Stop Loss: $0.39 | Take Profit: $0.46\n\n"
                
                "*Total Unrealized P/L:* +0.85%"
            )
            
            self.send_message(chat_id, open_positions_message)
            
        except Exception as e:
            logger.error(f"Error handling open command: {str(e)}", exc_info=True)
            
    def _handle_stats_command(self, message) -> None:
        """
        Handle /stats command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual stats
            stats_message = (
                "📊 *Detailed Performance Metrics*\n\n"
                "*Trade Statistics:*\n"
                "• Total Trades: *132*\n"
                "• Winning Trades: *90 (68.2%)*\n"
                "• Losing Trades: *42 (31.8%)*\n"
                "• Average Win: *3.5%*\n"
                "• Average Loss: *-1.9%*\n"
                "• Largest Win: *12.4%*\n"
                "• Largest Loss: *-5.2%*\n\n"
                
                "*Risk Metrics:*\n"
                "• Profit Factor: *1.85*\n"
                "• Sharpe Ratio: *1.73*\n"
                "• Sortino Ratio: *2.54*\n"
                "• Max Drawdown: *5.7%*\n"
                "• Recovery Factor: *4.12*\n\n"
                
                "*Portfolio Growth:*\n"
                "• Starting Capital: *$10,000.00*\n"
                "• Current Value: *$12,350.00*\n"
                "• Total Return: *+23.5%*\n"
                "• Monthly Return: *+3.5%*\n"
                "• Annual Return: *+42.0%*"
            )
            
            self.send_message(chat_id, stats_message)
            
        except Exception as e:
            logger.error(f"Error handling stats command: {str(e)}", exc_info=True)
            
    def _handle_report_command(self, message) -> None:
        """
        Handle /report command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Generate actual report
            report_message = (
                "📋 *Performance Report*\n\n"
                "I'm generating a comprehensive performance report for your account.\n\n"
                "This may take a few moments. I'll send you the report as soon as it's ready.\n\n"
                "The report will include:\n"
                "• Detailed performance metrics\n"
                "• Trade history analysis\n"
                "• Strategy performance breakdown\n"
                "• Risk analysis\n"
                "• Recommendations for improvement"
            )
            
            self.send_message(chat_id, report_message)
            
            # Simulate report generation (in a real bot, this would be done asynchronously)
            time.sleep(2)
            
            self.send_message(
                chat_id,
                "📊 Here's your performance report! You can download the full report as a PDF from our web dashboard."
            )
            
        except Exception as e:
            logger.error(f"Error handling report command: {str(e)}", exc_info=True)
            
    def _handle_pairs_command(self, message) -> None:
        """
        Handle /pairs command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual trading pairs
            pairs_message = (
                "🔄 *Trading Pairs*\n\n"
                "*Active Pairs:*\n"
                "✅ BTC/USDT\n"
                "✅ ETH/USDT\n"
                "✅ BNB/USDT\n"
                "✅ XRP/USDT\n"
                "✅ ADA/USDT\n\n"
                
                "*Inactive Pairs:*\n"
                "❌ SOL/USDT\n"
                "❌ DOT/USDT\n"
                "❌ DOGE/USDT\n"
                "❌ LINK/USDT\n"
                "❌ LTC/USDT\n\n"
                
                "To enable or disable a pair, use:\n"
                "/pair enable [symbol] or /pair disable [symbol]"
            )
            
            # Create inline keyboard for pair management
            markup = types.InlineKeyboardMarkup(row_width=2)
            markup.add(
                types.InlineKeyboardButton("Enable Pair", callback_data="pair_enable"),
                types.InlineKeyboardButton("Disable Pair", callback_data="pair_disable"),
                types.InlineKeyboardButton("Enable All", callback_data="pair_enable_all"),
                types.InlineKeyboardButton("Disable All", callback_data="pair_disable_all")
            )
            
            self.send_message(chat_id, pairs_message, reply_markup=markup)
            
        except Exception as e:
            logger.error(f"Error handling pairs command: {str(e)}", exc_info=True)
            
    def _handle_strategies_command(self, message) -> None:
        """
        Handle /strategies command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual strategies
            strategies_message = (
                "🧠 *Trading Strategies*\n\n"
                "*Active Strategies:*\n"
                "• Trend Following: *40%* allocation\n"
                "• Mean Reversion: *30%* allocation\n"
                "• Grid Trading: *20%* allocation\n"
                "• Scalping: *10%* allocation\n\n"
                
                "*Inactive Strategies:*\n"
                "• Sentiment-based Trading\n"
                "• Arbitrage\n\n"
                
                "To adjust strategy allocation, use:\n"
                "/strategy adjust [name] [percentage]"
            )
            
            # Create inline keyboard for strategy management
            markup = types.InlineKeyboardMarkup(row_width=2)
            markup.add(
                types.InlineKeyboardButton("Enable Strategy", callback_data="strategy_enable"),
                types.InlineKeyboardButton("Disable Strategy", callback_data="strategy_disable"),
                types.InlineKeyboardButton("Adjust Allocation", callback_data="strategy_adjust")
            )
            
            self.send_message(chat_id, strategies_message, reply_markup=markup)
            
        except Exception as e:
            logger.error(f"Error handling strategies command: {str(e)}", exc_info=True)
            
    def _handle_alerts_command(self, message) -> None:
        """
        Handle /alerts command
        """
        try:
            chat_id = message.chat.id
            
            # TODO: Get actual alert settings
            alerts_message = (
                "🔔 *Notification Settings*\n\n"
                "*Current Settings:*\n"
                "✅ Trade Execution\n"
                "✅ Take Profit Hit\n"
                "✅ Stop Loss Hit\n"
                "✅ Position Adjustment\n"
                "✅ Daily Summary\n"
                "❌ System Alerts\n\n"
                
                "To toggle an alert type, use:\n"
                "/alert toggle [type]"
            )
            
            # Create inline keyboard for alert management
            markup = types.InlineKeyboardMarkup(row_width=2)
            markup.add(
                types.InlineKeyboardButton("Trade Execution", callback_data="alert_trade_execution"),
                types.InlineKeyboardButton("Take Profit Hit", callback_data="alert_take_profit"),
                types.InlineKeyboardButton("Stop Loss Hit", callback_data="alert_stop_loss"),
                types.InlineKeyboardButton("Position Adjustment", callback_data="alert_position"),
                types.InlineKeyboardButton("Daily Summary", callback_data="alert_daily"),
                types.InlineKeyboardButton("System Alerts", callback_data="alert_system")
            )
            
            self.send_message(chat_id, alerts_message, reply_markup=markup)
            
        except Exception as e:
            logger.error(f"Error handling alerts command: {str(e)}", exc_info=True)
            
    def _handle_admin_command(self, message) -> None:
        """
        Handle /admin command
        """
        try:
            chat_id = message.chat.id
            
            # Check if user is admin
            if chat_id not in self.admin_chat_ids:
                self.send_message(chat_id, "⛔ You do not have permission to use admin commands.")
                return
                
            admin_message = (
                "👑 *Admin Commands*\n\n"
                "*User Management:*\n"
                "/users - List all registered users\n"
                "/user [email] - Show specific user details\n"
                "/pause_user [email] - Stop trading for user\n"
                "/resume_user [email] - Restart user trading\n"
                "/risk_user [email] [level] - Adjust user risk\n\n"
                
                "*System Control:*\n"
                "/status - Show overall system health\n"
                "/performance - Display aggregated results\n"
                "/pause_all - Halt all trading system-wide\n"
                "/resume_all - Restart all trading\n"
                "/close_all - Close all open positions\n\n"
                
                "*Technical Management:*\n"
                "/force_update - Initiate system update\n"
                "/restart_bot - Reboot the entire system\n"
                "/check_health - Run diagnostics\n"
                "/backup_now - Force immediate backup\n"
                "/logs - Retrieve system logs\n\n"
                
                "*Strategy Management:*\n"
                "/strategies - Show active strategies\n"
                "/enable [strategy] - Activate specific strategy\n"
                "/disable [strategy] - Deactivate strategy\n"
                "/backtest [strategy] - Run historical test\n"
                "/optimize [strategy] - Trigger re-optimization"
            )
            
            self.send_message(chat_id, admin_message)
            
        except Exception as e:
            logger.error(f"Error handling admin command: {str(e)}", exc_info=True)
            
    def _handle_admin_check_command(self, message) -> None:
        """
        Check if user is an admin
        """
        try:
            chat_id = message.chat.id
            user_id = message.from_user.id
            
            logger.info(f"Admin check from chat_id: {chat_id}, user_id: {user_id}")
            logger.info(f"Current admin_chat_ids: {self.admin_chat_ids}")
            
            if chat_id in self.admin_chat_ids:
                self.send_message(chat_id, "✅ You are registered as an admin.")
            else:
                self.send_message(chat_id, f"❌ You are not registered as an admin. Your chat_id is: {chat_id}")
                
        except Exception as e:
            logger.error(f"Error handling admin check command: {str(e)}", exc_info=True)
            
    def _handle_data_command(self, message) -> None:
        """
        Handle /data command
        """
        try:
            chat_id = message.chat.id
            
            data_message = (
                "📊 *Market Data Analysis*\n\n"
                "*Current Market Analysis:*\n"
                "• BTC Trend: *Bullish*\n"
                "• ETH Trend: *Neutral*\n"
                "• Market Volatility: *Medium*\n"
                "• Market Sentiment: *Positive*\n\n"
                
                "*Key Technical Indicators:*\n"
                "• BTC RSI: *62.5 (Bullish)*\n"
                "• ETH RSI: *48.3 (Neutral)*\n"
                "• BTC MACD: *Bullish Crossover*\n"
                "• Market Fear & Greed: *65 (Greed)*\n\n"
                
                "*Recent Market Events:*\n"
                "• Bitcoin ETF inflows: *+$125M (24h)*\n"
                "• Major exchange volumes: *+12% (24h)*\n"
                "• Top gainers: *SOL (+8.5%), DOT (+6.2%)*\n"
                "• Top losers: *XRP (-3.2%), LTC (-2.1%)*\n\n"
                
                "Use /pairs to manage your trading pairs or /performance to check your results."
            )
            
            self.send_message(chat_id, data_message)
            
        except Exception as e:
            logger.error(f"Error handling data command: {str(e)}", exc_info=True)
            
    def _handle_setup_exchange_command(self, message) -> None:
        """
        Handle /setup_exchange command
        """
        try:
            chat_id = message.chat.id
            
            setup_message = (
                "🔑 *Exchange API Setup*\n\n"
                "To connect your Binance account, you'll need to provide your API credentials.\n\n"
                "1. Log in to your Binance account\n"
                "2. Go to API Management\n"
                "3. Create a new API key with trading permissions\n"
                "4. Copy the API key and secret\n\n"
                "Ready to proceed? Type 'yes' to continue or click the button below."
            )
            
            # Create inline keyboard for setup
            markup = types.InlineKeyboardMarkup()
            markup.add(
                types.InlineKeyboardButton("Set Up Exchange", callback_data="setup_exchange")
            )
            
            self.send_message(chat_id, setup_message, reply_markup=markup)
            
            # Set user state
            self.user_states[chat_id] = {
                'state': 'awaiting_setup_confirmation',
                'data': {}
            }
            
        except Exception as e:
            logger.error(f"Error handling setup exchange command: {str(e)}", exc_info=True)
    
    def get_user_by_telegram_id(self, telegram_id: str) -> Optional[Any]:
        """
        Get user by Telegram ID
        
        Args:
            telegram_id: Telegram ID
            
        Returns:
            User object or None if not found
        """
        if self.user_repository:
            try:
                return self.user_repository.get_user_by_telegram_id(telegram_id)
            except Exception as e:
                logger.error(f"Error getting user by Telegram ID: {str(e)}", exc_info=True)
                return None
        return None
    
    def _handle_text_message(self, message) -> None:
        """
        Handle text messages for conversation flow
        """
        try:
            chat_id = message.chat.id
            text = message.text
            
            # Log the received message for debugging
            logger.debug(f"Received text message from {chat_id}: '{text}'")
            
            # Check if user is in a specific state
            if chat_id in self.user_states:
                state = self.user_states[chat_id]['state']
                data = self.user_states[chat_id]['data']
                
                if state == 'awaiting_email':
                    # Validate email
                    if '@' in text and '.' in text:
                        data['email'] = text
                        self.user_states[chat_id]['state'] = 'awaiting_confirmation'
                        
                        confirm_message = (
                            f"📧 Email address received: *{text}*\n\n"
                            f"Is this correct? (yes/no)"
                        )
                        
                        self.send_message(chat_id, confirm_message)
                    else:
                        self.send_message(chat_id, "❌ Invalid email format. Please enter a valid email address:")
                        
                elif state == 'awaiting_confirmation':
                    if text.lower() in ['yes', 'y']:
                        # Process registration
                        email = data.get('email', '')
                        telegram_id = str(message.from_user.id)
                        
                        # Register user in database if user repository available
                        user_id = None
                        if self.user_repository:
                            try:
                                user = self.user_repository.create_user(
                                    telegram_id=telegram_id,
                                    email=email,
                                    first_name=message.from_user.first_name,
                                    last_name=message.from_user.last_name,
                                    username=message.from_user.username
                                )
                                if user:
                                    user_id = user.id
                                    data['user_id'] = user_id
                            except Exception as e:
                                logger.error(f"Error creating user: {str(e)}", exc_info=True)
                        
                        # Update success message to continue with API key setup
                        success_message = (
                            "✅ *Registration Step 1 Complete!*\n\n"
                            f"Your account has been created with email: *{email}*\n\n"
                            f"Let's set up your exchange API keys now.\n\n"
                            f"Please enter your Binance API key:"
                        )
                        
                        self.send_message(chat_id, success_message)
                        
                        # Update state to await API key
                        self.user_states[chat_id]['state'] = 'awaiting_api_key'
                        
                    elif text.lower() in ['no', 'n']:
                        # Restart registration
                        self.user_states[chat_id]['state'] = 'awaiting_email'
                        self.user_states[chat_id]['data'] = {}
                        
                        self.send_message(chat_id, "Please enter your email address:")
                        
                    else:
                        self.send_message(chat_id, "Please respond with 'yes' or 'no':")
                
                elif state == 'awaiting_api_key':
                    # Store API key
                    data['api_key'] = text
                    self.user_states[chat_id]['state'] = 'awaiting_api_secret'
                    
                    self.send_message(chat_id, "Great! Now please enter your Binance API secret:")
                    
                elif state == 'awaiting_api_secret':
                    # Store API secret
                    data['api_secret'] = text
                    
                    # Store the API credentials securely if API key repository available
                    success = False
                    if self.api_key_repository and 'user_id' in data:
                        try:
                            key_id = self.api_key_repository.save_api_key(
                                user_id=data['user_id'],
                                exchange="binance",
                                api_key=data['api_key'],
                                api_secret=data['api_secret']
                            )
                            success = bool(key_id)
                        except Exception as e:
                            logger.error(f"Error saving API key: {str(e)}", exc_info=True)
                    
                    # Customize message based on success
                    if success:
                        status_message = "🎉 *API Keys Saved Successfully!*\n\n"
                    else:
                        status_message = "⚠️ *Note: API Keys couldn't be securely stored at this time*\n\n"
                    
                    complete_message = (
                        f"{status_message}"
                        "Your trading bot is now configured with your Binance account.\n\n"
                        "• Use /status to check your account status\n"
                        "• Use /risk to set your risk level\n"
                        "• Use /mode to choose between paper and live trading\n"
                        "• Use /pairs to select trading pairs\n\n"
                        "Happy trading! 📈"
                    )
                    
                    self.send_message(chat_id, complete_message)
                    
                    # Clear user state
                    del self.user_states[chat_id]
                
                elif state == 'awaiting_setup_confirmation':
                    if text.lower() in ['yes', 'y']:
                        self.send_message(chat_id, "Please enter your Binance API key:")
                        self.user_states[chat_id]['state'] = 'awaiting_api_key'
                    else:
                        self.send_message(chat_id, "You can set up your exchange connection later using the /setup_exchange command.")
                        del self.user_states[chat_id]
                
                elif state == 'awaiting_pair_enable':
                    # Process pair enable request
                    symbol = text.strip().upper()
                    # TODO: Enable trading pair
                    self.send_message(chat_id, f"✅ Trading pair *{symbol}* has been enabled.")
                    # Clear user state
                    del self.user_states[chat_id]
                
                elif state == 'awaiting_pair_disable':
                    # Process pair disable request
                    symbol = text.strip().upper()
                    # TODO: Disable trading pair
                    self.send_message(chat_id, f"❌ Trading pair *{symbol}* has been disabled.")
                    # Clear user state
                    del self.user_states[chat_id]
                
                elif state == 'awaiting_strategy_adjustment':
                    # Process strategy adjustment
                    # TODO: Parse input and adjust strategy
                    self.send_message(chat_id, "Strategy allocation has been updated.")
                    # Clear user state
                    del self.user_states[chat_id]
                
            else:
                # Check if message is "data" which caused errors in logs
                if text.lower() == "data":
                    self._handle_data_command(message)
                else:
                    # Default response for unsolicited messages
                    help_message = (
                        "I'm not sure what you're asking. Here are some commands you can use:\n\n"
                        "/help - Show all available commands\n"
                        "/status - Check your account status\n"
                        "/register - Create a new account"
                    )
                    
                    self.send_message(chat_id, help_message)
                
        except Exception as e:
            logger.error(f"Error handling text message: '{message.text}'", exc_info=True)
            
    def _process_message_queue(self) -> None:
        """
        Process message queue with rate limiting
        """
        while self.running:
            try:
                # Get message from queue
                try:
                    message_data = self.message_queue.get(timeout=1)
                except queue.Empty:
                    continue
                    
                # Send message
                chat_id = message_data.get('chat_id')
                text = message_data.get('text', '')
                reply_markup = message_data.get('reply_markup')
                
                if chat_id and text:
                    try:
                        if reply_markup:
                            self.bot.send_message(
                                chat_id,
                                text,
                                parse_mode='Markdown',
                                reply_markup=reply_markup
                            )
                        else:
                            self.bot.send_message(
                                chat_id,
                                text,
                                parse_mode='Markdown'
                            )
                    except ApiTelegramException as e:
                        if e.error_code == 429:
                            # Rate limit exceeded, wait and retry
                            retry_after = e.result_json.get('parameters', {}).get('retry_after', 5)
                            logger.warning(f"Rate limit exceeded, waiting {retry_after} seconds")
                            time.sleep(retry_after)
                            
                            # Put message back in queue
                            self.message_queue.put(message_data)
                        else:
                            logger.error(f"Telegram API error: {str(e)}", exc_info=True)
                    except Exception as e:
                        logger.error(f"Error sending Telegram message: {str(e)}", exc_info=True)
                        
                # Mark as done
                self.message_queue.task_done()
                
                # Sleep to avoid rate limiting
                time.sleep(0.05)
                
            except Exception as e:
                logger.error(f"Error processing message queue: {str(e)}", exc_info=True)
                
    def send_message(self, chat_id: int, text: str, reply_markup=None) -> None:
        """
        Send message to user via queue
        
        Args:
            chat_id: Chat ID
            text: Message text
            reply_markup: Optional reply markup
        """
        try:
            # Queue message
            self.message_queue.put({
                'chat_id': chat_id,
                'text': text,
                'reply_markup': reply_markup
            })
            
        except Exception as e:
            logger.error(f"Error queuing message: {str(e)}", exc_info=True)
            
    def send_message_to_admins(self, text: str) -> None:
        """
        Send message to all admin users
        
        Args:
            text: Message text
        """
        try:
            for admin_chat_id in self.admin_chat_ids:
                self.send_message(admin_chat_id, text)
                
        except Exception as e:
            logger.error(f"Error sending message to admins: {str(e)}", exc_info=True)
            
    def register_command_handler(self, command: str, handler: Callable) -> None:
        """
        Register custom command handler
        
        Args:
            command: Command name (without slash)
            handler: Handler function
        """
        try:
            self.command_handlers[command] = handler
            
            # Register with Telegram if bot is running
            if self.bot:
                @self.bot.message_handler(commands=[command])
                def handle_command(message):
                    handler(message)
                    
        except Exception as e:
            logger.error(f"Error registering command handler: {str(e)}", exc_info=True)
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get the status of the Telegram bot
        
        Returns:
            Dict with status information
        """
        return {
            "running": self.running,
            "message_queue_size": self.message_queue.qsize() if self.message_queue else 0,
            "active_conversations": len(self.user_states),
            "last_error": None  # TODO: Track last error
        }
        
    def restart(self) -> None:
        """
        Restart the Telegram bot
        """
        logger.info("Restarting Telegram bot")
        self.stop()
        time.sleep(2)  # Give it time to properly shut down
        self.start()
        logger.info("Telegram bot restarted")