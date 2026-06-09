from langchain_core.tools import tool

VERBOSE_DESCRIPTION = (
    "Send an email to one or more recipients. Provide the recipient email addresses "
    "(comma-separated for multiple), subject line, and body content. The email will "
    "be sent from the system's configured email account.\n\n"
    "The body supports plain text and basic markdown formatting. For HTML emails, "
    "wrap the body in <html> tags. Attachments are not currently supported — "
    "reference file paths or URLs instead.\n\n"
    "Common use cases: sending analysis reports, status updates, alerts, "
    "meeting summaries, or data export notifications.\n\n"
    "Example: send_email(\n"
    "  to='team@company.com',\n"
    "  subject='Q4 Analysis Complete',\n"
    "  body='The quarterly analysis is ready. Key findings...'\n"
    ")\n\n"
    "Note: Rate limited to 10 emails per minute. Emails to external domains "
    "require approval. All emails are logged for compliance."
)

SEND_EMAIL_PRUNED = "Send an email with specified recipients, subject, and body."

SEND_EMAIL_ROUTING = "Send an email."


@tool(description=VERBOSE_DESCRIPTION)
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email."""
    recipients = [r.strip() for r in to.split(",")]
    return (
        f"Email sent successfully.\n"
        f"  To: {', '.join(recipients)}\n"
        f"  Subject: {subject}\n"
        f"  Body length: {len(body)} characters\n"
        f"  Message ID: <mock-{hash(subject) % 10000:04d}@company.com>"
    )


send_email_tool = send_email
