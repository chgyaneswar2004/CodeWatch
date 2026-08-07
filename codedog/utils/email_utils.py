import os
import smtplib
import ssl
import html
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import List, Optional

from codedog.config.settings import settings



class EmailNotifier:
    """Email notification utility for sending code review reports."""
    
    def __init__(
        self,
        smtp_server: str = None,
        smtp_port: int = None,
        smtp_username: str = None,
        smtp_password: str = None,
        use_tls: bool = True,
    ):
        """Initialize EmailNotifier with SMTP settings.
        
        Args:
            smtp_server: SMTP server address (defaults to env var SMTP_SERVER)
            smtp_port: SMTP server port (defaults to env var SMTP_PORT)
            smtp_username: SMTP username (defaults to env var SMTP_USERNAME)
            smtp_password: SMTP password (defaults to env var SMTP_PASSWORD)
            use_tls: Whether to use TLS for SMTP connection (defaults to True)
        """
        self.smtp_server = smtp_server or settings.smtp_server
        self.smtp_port = smtp_port or settings.smtp_port
        self.smtp_username = smtp_username or settings.smtp_username
        self.smtp_password = smtp_password or settings.smtp_password
        self.use_tls = use_tls
        
        # Validate required settings
        if not all([self.smtp_server, self.smtp_username, self.smtp_password]):
            missing = []
            if not self.smtp_server:
                missing.append("SMTP_SERVER")
            if not self.smtp_username:
                missing.append("SMTP_USERNAME")
            if not self.smtp_password:
                missing.append("SMTP_PASSWORD")
            
            raise ValueError(f"Missing required email configuration: {', '.join(missing)}")
    
    def send_report(
        self,
        to_emails: List[str],
        subject: str,
        markdown_content: str,
        from_email: Optional[str] = None,
        cc_emails: Optional[List[str]] = None,
    ) -> bool:
        """Send code review report as email.
        
        Args:
            to_emails: List of recipient email addresses
            subject: Email subject
            markdown_content: Report content in markdown format
            from_email: Sender email (defaults to SMTP_USERNAME)
            cc_emails: List of CC email addresses
            
        Returns:
            bool: True if email was sent successfully, False otherwise
        """
        if not to_emails:
            raise ValueError("No recipient emails provided")
        
        # Create message
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = from_email or self.smtp_username
        msg["To"] = ", ".join(to_emails)
        
        if cc_emails:
            msg["Cc"] = ", ".join(cc_emails)
            all_recipients = to_emails + cc_emails
        else:
            all_recipients = to_emails
        
        # Attach markdown content as both plain text and HTML
        text_part = MIMEText(markdown_content, "plain")
        
        # Escape markdown content to prevent HTML/script injection
        escaped_content = html.escape(markdown_content)
        html_content = f"""
        <html>
            <head>
                <style>
                    body {{
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
                        color: #24292f;
                        background-color: #f6f8fa;
                        margin: 0;
                        padding: 20px;
                    }}
                    .container {{
                        max-width: 800px;
                        margin: 0 auto;
                        background-color: #ffffff;
                        padding: 30px;
                        border-radius: 6px;
                        border: 1px solid #d0d7de;
                        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
                    }}
                    pre {{
                        white-space: pre-wrap;
                        word-wrap: break-word;
                        font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
                        font-size: 13.6px;
                        line-height: 1.45;
                        background-color: #f6f8fa;
                        padding: 16px;
                        border-radius: 6px;
                        border: 1px solid #d0d7de;
                        color: #24292f;
                    }}
                </style>
            </head>
            <body>
                <div class="container">
                    <pre>{escaped_content}</pre>
                </div>
            </body>
        </html>
        """
        html_part = MIMEText(html_content, "html")
        
        msg.attach(text_part)
        msg.attach(html_part)
        
        try:
            # Create a secure SSL context
            context = ssl.create_default_context() if self.use_tls else None
            
            with smtplib.SMTP(self.smtp_server, self.smtp_port, timeout=10) as server:
                if self.use_tls:
                    server.starttls(context=context)
                
                server.login(self.smtp_username, self.smtp_password)
                server.sendmail(
                    self.smtp_username, all_recipients, msg.as_string()
                )
            
            return True
        except Exception as e:
            print(f"Failed to send email: {str(e)}")
            return False


def send_report_email(
    to_emails: List[str],
    subject: str,
    markdown_content: str,
    cc_emails: Optional[List[str]] = None,
) -> bool:
    """Helper function to send code review report via email.
    
    Args:
        to_emails: List of recipient email addresses
        subject: Email subject
        markdown_content: Report content in markdown format
        cc_emails: List of CC email addresses
            
    Returns:
        bool: True if email was sent successfully, False otherwise
    """
    # Check if email notification is enabled
    if not settings.email_enabled:
        print("Email notifications are disabled. Set EMAIL_ENABLED=true to enable.")
        return False
    
    try:
        notifier = EmailNotifier()
        return notifier.send_report(
            to_emails=to_emails,
            subject=subject,
            markdown_content=markdown_content,
            cc_emails=cc_emails,
        )
    except ValueError as e:
        print(f"Email configuration error: {str(e)}")
        return False
    except smtplib.SMTPAuthenticationError:
        print("SMTP Authentication Error: Invalid username or password.")
        print("If using Gmail, make sure to:")
        print("1. Enable 2-step verification for your Google account")
        print("2. Generate an App Password at https://myaccount.google.com/apppasswords")
        print("3. Use that App Password in your .env file, not your regular Gmail password")
        return False
    except Exception as e:
        print(f"Unexpected error sending email: {str(e)}")
        return False 