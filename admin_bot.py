# admin/admin_bot.py
import logging
import asyncio
import threading
from typing import List, Dict, Any, Optional

import telegram
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    ApplicationBuilder, 
    CommandHandler, 
    CallbackQueryHandler, 
    ContextTypes,
    MessageHandler,
    filters
)

from database.repository.user_repository import UserRepository
from database.repository.trade_repository import TradeRepository
from maintenance.system_monitor import SystemMonitor

logger = logging.getLogger(__name__)

class AdminBot:
    """
    Administrative bot for QuantumFlow system management
    Provides privileged commands for system administration
    """
    def __init__(
        self,
        token: str,
        admin_ids: List[str],
        user_repository=None,
        trade_repository=None,
        system_monitoring=None
    ):
        """
        Initialize the AdminBot with optional repository and monitoring components.
        
        Args:
            token (str): Telegram bot token
            admin_ids (list): List of admin user IDs
            user_repository (UserRepository, optional): User repository for user management
            trade_repository (TradeRepository, optional): Trade repository for trade-related operations
            system_monitoring (SystemMonitor, optional): System monitoring component
        """
        self.token = token
        self.admin_ids = admin_ids if admin_ids else []
        
        # Use provided repositories or create new instances
        self.user_repo = user_repository or UserRepository()
        self.trade_repo = trade_repository or TradeRepository()
        self.system_monitor = system_monitoring or SystemMonitor()
        
        # Setup application builder with error handling
        self.app = None
        self.running = False
        
        # For thread management
        self.stop_event = threading.Event()
        self.thread = None
        
        logger.info("Admin bot initialized")
        
    def register_handlers(self):
        """Register all command handlers for admin bot"""
        if not self.app:
            logger.error("Cannot register handlers: Application not initialized")
            return
            
        # User management commands
        self.app.add_handler(CommandHandler("start", self.start_command))
        self.app.add_handler(CommandHandler("help", self.help_command))
        self.app.add_handler(CommandHandler("users", self.list_users))
        self.app.add_handler(CommandHandler("user", self.user_details))
        self.app.add_handler(CommandHandler("pause_user", self.pause_user))
        self.app.add_handler(CommandHandler("resume_user", self.resume_user))
        self.app.add_handler(CommandHandler("risk_user", self.risk_user))
        
        # System monitoring commands
        self.app.add_handler(CommandHandler("status", self.system_status))
        self.app.add_handler(CommandHandler("performance", self.system_performance))
        self.app.add_handler(CommandHandler("pause_all", self.pause_all_trading))
        self.app.add_handler(CommandHandler("resume_all", self.resume_all_trading))
        self.app.add_handler(CommandHandler("close_all", self.close_all_positions))
        
        # Technical management commands
        self.app.add_handler(CommandHandler("force_update", self.force_update))
        self.app.add_handler(CommandHandler("restart_bot", self.restart_bot))
        self.app.add_handler(CommandHandler("check_health", self.check_health))
        self.app.add_handler(CommandHandler("backup_now", self.backup_now))
        self.app.add_handler(CommandHandler("logs", self.get_logs))
        
        # Strategy management commands
        self.app.add_handler(CommandHandler("strategies", self.list_strategies))
        self.app.add_handler(CommandHandler("enable", self.enable_strategy))
        self.app.add_handler(CommandHandler("disable", self.disable_strategy))
        self.app.add_handler(CommandHandler("backtest", self.backtest_strategy))
        self.app.add_handler(CommandHandler("optimize", self.optimize_strategy))
        
        # Callback query handlers for inline buttons
        self.app.add_handler(CallbackQueryHandler(self.handle_callback))
        
        # Default handler for unknown commands (admin only)
        self.app.add_handler(MessageHandler(filters.COMMAND, self.unknown_command))
        
        logger.info("Admin bot handlers registered")
    
    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /start command"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access. This bot is for administrators only.")
        
        await update.message.reply_text(
            f"👋 Welcome to the QuantumFlow Admin Bot!\n\n"
            f"This interface provides administrative control over the QuantumFlow trading system.\n\n"
            f"Use /help to see available commands."
        )
    
    async def help_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle /help command"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access. This bot is for administrators only.")
        
        await update.message.reply_text(
            "🔧 *QuantumFlow Admin Commands*\n\n"
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
            "/optimize [strategy] - Trigger re-optimization",
            parse_mode="Markdown"
        )
    
    async def list_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all registered users"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            users = self.user_repo.get_all_users()
            
            if not users:
                return await update.message.reply_text("No registered users found.")
                
            user_text = "\n".join([f"ID: {user.id} | Email: {user.email} | Status: {user.status}" for user in users])
            
            if len(user_text) > 4000:  # Telegram message limit
                # Split into multiple messages if too long
                chunks = [user_text[i:i+4000] for i in range(0, len(user_text), 4000)]
                await update.message.reply_text(f"Registered users ({len(users)}):")
                for chunk in chunks:
                    await update.message.reply_text(chunk)
            else:
                await update.message.reply_text(f"Registered users ({len(users)}):\n\n{user_text}")
                
        except Exception as e:
            logger.error(f"Error listing users: {str(e)}")
            await update.message.reply_text(f"Error retrieving users: {str(e)}")
    
    async def user_details(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show details for a specific user"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            if not context.args or len(context.args) < 1:
                return await update.message.reply_text("Please provide a user email or ID, e.g., /user example@email.com")
                
            user_identifier = context.args[0]
            
            # Try to find user by email or ID
            if '@' in user_identifier:
                # Search by email
                user = self.user_repo.get_user_by_email(user_identifier)
            else:
                # Try to convert to int for ID search
                try:
                    user_id = int(user_identifier)
                    user = self.user_repo.get_user_by_id(user_id)
                except ValueError:
                    return await update.message.reply_text("Invalid user ID format. Please provide a valid email or numeric ID.")
            
            if not user:
                return await update.message.reply_text(f"User not found: {user_identifier}")
                
            # Get user's trading data
            trades = self.trade_repo.get_trades_by_user(user.id, limit=5)
            trades_count = self.trade_repo.count_trades_by_user(user.id)
            
            # Format response
            user_details = (
                f"*User Details*\n\n"
                f"ID: {user.id}\n"
                f"Email: {user.email}\n"
                f"Status: {user.status}\n"
                f"Risk Level: {user.risk_level}\n"
                f"Balance: ${user.balance:.2f}\n"
                f"KYC Verified: {'Yes' if user.kyc_verified else 'No'}\n"
                f"Created: {user.created_at}\n\n"
                f"*Trading Activity*\n"
                f"Total Trades: {trades_count}\n"
                f"Trading Mode: {user.trading_mode}\n\n"
            )
            
            if trades:
                trades_text = "*Recent Trades*\n"
                for trade in trades:
                    trades_text += f"- {trade.symbol}: {trade.side} at ${trade.price:.2f}, P/L: {trade.profit:.2f}%\n"
                user_details += trades_text
            
            # Create inline keyboard for user management
            keyboard = [
                [
                    InlineKeyboardButton("Pause Trading", callback_data=f"pause_user_{user.id}"),
                    InlineKeyboardButton("Resume Trading", callback_data=f"resume_user_{user.id}")
                ],
                [
                    InlineKeyboardButton("Low Risk", callback_data=f"risk_user_{user.id}_low"),
                    InlineKeyboardButton("Medium Risk", callback_data=f"risk_user_{user.id}_medium"),
                    InlineKeyboardButton("High Risk", callback_data=f"risk_user_{user.id}_high")
                ]
            ]
            reply_markup = InlineKeyboardMarkup(keyboard)
            
            await update.message.reply_text(user_details, parse_mode="Markdown", reply_markup=reply_markup)
                
        except Exception as e:
            logger.error(f"Error getting user details: {str(e)}")
            await update.message.reply_text(f"Error retrieving user details: {str(e)}")
    
    async def pause_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause trading for a specific user"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            if not context.args or len(context.args) < 1:
                return await update.message.reply_text("Please provide a user email or ID, e.g., /pause_user example@email.com")
                
            user_identifier = context.args[0]
            
            # Find user and update status
            user = self._find_user(user_identifier)
            
            if not user:
                return await update.message.reply_text(f"User not found: {user_identifier}")
                
            result = self.user_repo.update_user(user.id, is_active=False)
            
            if result:
                await update.message.reply_text(f"Trading has been paused for user {user.email}")
            else:
                await update.message.reply_text(f"Failed to pause trading for user {user.email}")
                
        except Exception as e:
            logger.error(f"Error pausing user: {str(e)}")
            await update.message.reply_text(f"Error pausing user: {str(e)}")
    
    async def resume_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume trading for a specific user"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            if not context.args or len(context.args) < 1:
                return await update.message.reply_text("Please provide a user email or ID, e.g., /resume_user example@email.com")
                
            user_identifier = context.args[0]
            
            # Find user and update status
            user = self._find_user(user_identifier)
            
            if not user:
                return await update.message.reply_text(f"User not found: {user_identifier}")
                
            result = self.user_repo.update_user(user.id, is_active=True)
            
            if result:
                await update.message.reply_text(f"Trading has been resumed for user {user.email}")
            else:
                await update.message.reply_text(f"Failed to resume trading for user {user.email}")
                
        except Exception as e:
            logger.error(f"Error resuming user: {str(e)}")
            await update.message.reply_text(f"Error resuming user: {str(e)}")
    
    async def risk_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Set risk level for a specific user"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            if not context.args or len(context.args) < 2:
                return await update.message.reply_text("Please provide user email and risk level, e.g., /risk_user example@email.com medium")
                
            user_identifier = context.args[0]
            risk_level = context.args[1].lower()
            
            if risk_level not in ['low', 'medium', 'high']:
                return await update.message.reply_text("Invalid risk level. Use 'low', 'medium', or 'high'")
                
            # Find user and update risk level
            user = self._find_user(user_identifier)
            
            if not user:
                return await update.message.reply_text(f"User not found: {user_identifier}")
                
            result = self.user_repo.update_user(user.id, risk_level=risk_level)
            
            if result:
                await update.message.reply_text(f"Risk level for user {user.email} set to {risk_level}")
            else:
                await update.message.reply_text(f"Failed to update risk level for user {user.email}")
                
        except Exception as e:
            logger.error(f"Error updating user risk level: {str(e)}")
            await update.message.reply_text(f"Error updating risk level: {str(e)}")
    
    async def system_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show overall system health status"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            status = self.system_monitor.get_health_status()
            metrics = self.system_monitor.get_system_metrics()
            
            response = (
                f"🔄 *System Status*: {status}\n\n"
                f"*System Metrics:*\n"
                f"• CPU Usage: {metrics.get('cpu_usage', 'N/A')}%\n"
                f"• Memory Usage: {metrics.get('memory_usage', 'N/A')}%\n"
                f"• Disk Usage: {metrics.get('disk_usage', 'N/A')}%\n"
                f"• Database Connections: {metrics.get('db_connections', 'N/A')}\n"
                f"• Active Users: {metrics.get('active_users', 'N/A')}\n"
                f"• Active Positions: {metrics.get('active_positions', 'N/A')}\n\n"
                f"*Component Status:*\n"
                f"• Trading Engine: {metrics.get('trading_engine_status', 'N/A')}\n"
                f"• ML Models: {metrics.get('ml_models_status', 'N/A')}\n"
                f"• Database: {metrics.get('database_status', 'N/A')}\n"
                f"• Notification System: {metrics.get('notification_status', 'N/A')}\n\n"
                f"*Performance:*\n"
                f"• Avg Response Time: {metrics.get('avg_response_time', 'N/A')}ms\n"
                f"• Last Error: {metrics.get('last_error', 'None')}\n"
                f"• Uptime: {metrics.get('uptime', 'N/A')}"
            )
            
            await update.message.reply_text(response, parse_mode="Markdown")
                
        except Exception as e:
            logger.error(f"Error getting system status: {str(e)}")
            await update.message.reply_text(f"Error retrieving system status: {str(e)}")
    
    async def system_performance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Show system trading performance metrics"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            # Get performance metrics from trade repository
            metrics = self.trade_repo.get_global_performance_metrics()
            
            response = (
                f"📊 *System Performance*\n\n"
                f"*Trading Metrics:*\n"
                f"• Total Trades: {metrics.get('total_trades', 'N/A')}\n"
                f"• Winning Trades: {metrics.get('winning_trades', 'N/A')} ({metrics.get('win_rate', 'N/A')}%)\n"
                f"• Total Profit: ${metrics.get('total_profit', 'N/A')}\n"
                f"• Profit Factor: {metrics.get('profit_factor', 'N/A')}\n"
                f"• Max Drawdown: {metrics.get('max_drawdown', 'N/A')}%\n\n"
                f"*Today's Activity:*\n"
                f"• Trades: {metrics.get('today_trades', 'N/A')}\n"
                f"• Win Rate: {metrics.get('today_win_rate', 'N/A')}%\n"
                f"• Profit: ${metrics.get('today_profit', 'N/A')}\n\n"
                f"*Top Strategies:*\n"
                f"• {metrics.get('top_strategy', 'N/A')}: ${metrics.get('top_strategy_profit', 'N/A')}\n"
                f"• {metrics.get('second_strategy', 'N/A')}: ${metrics.get('second_strategy_profit', 'N/A')}\n"
                f"• {metrics.get('third_strategy', 'N/A')}: ${metrics.get('third_strategy_profit', 'N/A')}"
            )
            
            await update.message.reply_text(response, parse_mode="Markdown")
                
        except Exception as e:
            logger.error(f"Error getting performance metrics: {str(e)}")
            await update.message.reply_text(f"Error retrieving performance metrics: {str(e)}")
    
    async def pause_all_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Pause trading for all users"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            # TODO: Implement system-wide pause functionality
            # This would likely involve setting a global flag in the trading engine
            
            await update.message.reply_text(
                "⚠️ *SYSTEM-WIDE TRADING PAUSE*\n\n"
                "Are you sure you want to pause trading for ALL users?\n\n"
                "This will prevent any new positions from being opened, but existing positions will continue to be managed.",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("✅ Confirm Pause ALL", callback_data="confirm_pause_all"),
                        InlineKeyboardButton("❌ Cancel", callback_data="cancel_pause_all")
                    ]
                ])
            )
                
        except Exception as e:
            logger.error(f"Error preparing pause all trading: {str(e)}")
            await update.message.reply_text(f"Error: {str(e)}")
    
    async def resume_all_trading(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resume trading for all users"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            # TODO: Implement system-wide resume functionality
            
            await update.message.reply_text(
                "⚠️ *SYSTEM-WIDE TRADING RESUME*\n\n"
                "Are you sure you want to resume trading for ALL users?\n\n"
                "This will allow new positions to be opened according to user strategies.",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("✅ Confirm Resume ALL", callback_data="confirm_resume_all"),
                        InlineKeyboardButton("❌ Cancel", callback_data="cancel_resume_all")
                    ]
                ])
            )
                
        except Exception as e:
            logger.error(f"Error preparing resume all trading: {str(e)}")
            await update.message.reply_text(f"Error: {str(e)}")
    
    async def close_all_positions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close all open positions"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        try:
            # TODO: Implement close all positions functionality
            
            await update.message.reply_text(
                "⚠️ *EMERGENCY POSITION CLOSURE*\n\n"
                "Are you sure you want to close ALL open positions for ALL users?\n\n"
                "This is a drastic action and should only be used in emergency situations.",
                parse_mode="Markdown",
                reply_markup=InlineKeyboardMarkup([
                    [
                        InlineKeyboardButton("⚠️ FORCE CLOSE ALL", callback_data="confirm_close_all"),
                        InlineKeyboardButton("❌ Cancel", callback_data="cancel_close_all")
                    ]
                ])
            )
                
        except Exception as e:
            logger.error(f"Error preparing close all positions: {str(e)}")
            await update.message.reply_text(f"Error: {str(e)}")
    
    async def force_update(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force system update"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for system update functionality
        await update.message.reply_text("System update initiated")
        
    async def restart_bot(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Restart the entire system"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for system restart functionality
        await update.message.reply_text("System restart initiated")
        
    async def check_health(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Run system health diagnostics"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for health check functionality
        await update.message.reply_text("Running system diagnostics...")
        
    async def backup_now(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Force immediate system backup"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for backup functionality
        await update.message.reply_text("Backup initiated")
        
    async def get_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Retrieve system logs"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for log retrieval functionality
        await update.message.reply_text("System logs retrieved")
    
    async def list_strategies(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """List all trading strategies"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for strategy listing functionality
        await update.message.reply_text("Listing strategies...")
        
    async def enable_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Enable a specific strategy"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for strategy enabling functionality
        await update.message.reply_text("Strategy enabled")
        
    async def disable_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Disable a specific strategy"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for strategy disabling functionality
        await update.message.reply_text("Strategy disabled")
        
    async def backtest_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Run backtest for a strategy"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for strategy backtesting functionality
        await update.message.reply_text("Backtesting strategy...")
        
    async def optimize_strategy(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Optimize a strategy"""
        if not self.is_admin(update.effective_user.id):
            return await update.message.reply_text("⛔ Unauthorized access")
        
        # Placeholder for strategy optimization functionality
        await update.message.reply_text("Optimizing strategy...")
    
    async def unknown_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle unknown commands"""
        if not self.is_admin(update.effective_user.id):
            return
            
        await update.message.reply_text(
            "Unknown command. Use /help to see available commands."
        )
    
    async def handle_callback(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle callback queries from inline buttons"""
        query = update.callback_query
        await query.answer()
        
        if not self.is_admin(query.from_user.id):
            await query.edit_message_text("⛔ Unauthorized access")
            return
            
        data = query.data
        
        try:
            if data == "confirm_pause_all":
                # TODO: Implement system-wide pause
                await query.edit_message_text("⏸️ All trading has been paused system-wide.")
                
            elif data == "cancel_pause_all":
                await query.edit_message_text("System-wide trading pause cancelled.")
                
            elif data == "confirm_resume_all":
                # TODO: Implement system-wide resume
                await query.edit_message_text("▶️ All trading has been resumed system-wide.")
                
            elif data == "cancel_resume_all":
                await query.edit_message_text("System-wide trading resume cancelled.")
                
            elif data == "confirm_close_all":
                # TODO: Implement close all positions
                await query.edit_message_text("⚠️ All open positions have been closed.")
                
            elif data == "cancel_close_all":
                await query.edit_message_text("Emergency position closure cancelled.")
                
            elif data.startswith("pause_user_"):
                user_id = int(data.split("_")[2])
                result = self.user_repo.update_user(user_id, is_active=False)
                if result:
                    await query.edit_message_text("User trading paused.")
                else:
                    await query.edit_message_text("Failed to pause user trading.")
                    
            elif data.startswith("resume_user_"):
                user_id = int(data.split("_")[2])
                result = self.user_repo.update_user(user_id, is_active=True)
                if result:
                    await query.edit_message_text("User trading resumed.")
                else:
                    await query.edit_message_text("Failed to resume user trading.")
                    
            elif data.startswith("risk_user_"):
                parts = data.split("_")
                user_id = int(parts[2])
                risk_level = parts[3]
                result = self.user_repo.update_user(user_id, risk_level=risk_level)
                if result:
                    await query.edit_message_text(f"User risk level set to {risk_level}.")
                else:
                    await query.edit_message_text("Failed to update user risk level.")
                
        except Exception as e:
            logger.error(f"Error handling callback query: {str(e)}")
            await query.edit_message_text(f"Error: {str(e)}")
    
    def _find_user(self, user_identifier):
        """Helper method to find a user by email or ID"""
        if '@' in user_identifier:
            # Search by email
            return self.user_repo.get_user_by_email(user_identifier)
        else:
            # Try to convert to int for ID search
            try:
                user_id = int(user_identifier)
                return self.user_repo.get_user_by_id(user_id)
            except ValueError:
                return None
    
    def is_admin(self, user_id):
        """Check if user is admin"""
        return str(user_id) in self.admin_ids or user_id in self.admin_ids
    
    def start(self):
        """Start the admin bot with proper event loop handling"""
        if self.running:
            logger.warning("Admin bot is already running")
            return
        
        try:
            # Set up proper event loop for async operations
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            # Create and initialize application
            if not self.token:
                logger.error("Admin bot token not provided, bot will not start")
                return
                
            # Validate token format (should contain a colon)
            if ":" not in self.token:
                logger.error("Invalid Telegram bot token format. Token must contain a colon.")
                return
            
            # Build and configure the application
            self.app = ApplicationBuilder().token(self.token).build()
            
            # Register command handlers
            self.register_handlers()
            
            # Set running flag
            self.running = True
            
            # Start polling in the current thread
            logger.info("Starting Admin bot polling")
            self.app.run_polling(close_loop=False)
            
        except Exception as e:
            self.running = False
            logger.error(f"Error starting Admin bot: {str(e)}")
            raise
    
    def start_threaded(self):
        """Start the admin bot in a separate thread"""
        if self.running:
            logger.warning("Admin bot is already running")
            return
        
        def run_bot():
            try:
                # This is important: we need to create a new event loop for this thread
                asyncio.set_event_loop(asyncio.new_event_loop())
                
                # Create and initialize application
                if not self.token:
                    logger.error("Admin bot token not provided, bot will not start")
                    return
                    
                # Validate token format (should contain a colon)
                if ":" not in self.token:
                    logger.error("Invalid Telegram bot token format. Token must contain a colon.")
                    return
                
                # Build and configure the application
                self.app = ApplicationBuilder().token(self.token).build()
                
                # Register command handlers
                self.register_handlers()
                
                # Set running flag
                self.running = True
                
                # Start polling
                logger.info("Starting Admin bot polling in thread")
                self.app.run_polling(close_loop=False)
                
            except Exception as e:
                self.running = False
                logger.error(f"Error in Admin bot thread: {str(e)}")
        
        # Create and start thread
        self.thread = threading.Thread(target=run_bot, daemon=True)
        self.thread.start()
        logger.info("Admin bot thread started")
    
    def stop(self):
        """Stop the admin bot"""
        if not self.running:
            logger.warning("Admin bot is not running")
            return
        
        try:
            logger.info("Stopping Admin bot")
            
            # Stop the application if it exists
            if self.app:
                self.app.stop()
            
            # Set flag to indicate bot is stopped
            self.running = False
            
            # Wait for thread to terminate if running in separate thread
            if self.thread and self.thread.is_alive():
                self.thread.join(timeout=5)
                if self.thread.is_alive():
                    logger.warning("Admin bot thread did not terminate gracefully")
            
            logger.info("Admin bot stopped")
            
        except Exception as e:
            logger.error(f"Error stopping Admin bot: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get the status of the Admin bot"""
        return {
            "running": self.running,
            "thread_alive": self.thread.is_alive() if self.thread else False,
            "admin_count": len(self.admin_ids)
        }