import email
import base64
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from html import unescape


@dataclass
class EmailData:
    """Structured email data for phishing detection"""
    file_path: str
    subject: str
    sender: str
    sender_domain: str
    recipient: str
    date: str
    body_text: str
    body_html: str
    urls: List[str]
    attachments: List[str]
    headers: Dict[str, str]
    suspicious_headers: Dict[str, str]

    def to_dict(self) -> Dict:
        """Convert to dictionary for LangChain"""
        return {
            'file_path': self.file_path,
            'subject': self.subject,
            'sender': self.sender,
            'sender_domain': self.sender_domain,
            'recipient': self.recipient,
            'date': self.date,
            'body_text': self.body_text,
            'body_html': self.body_html,
            'urls': self.urls,
            'attachments': self.attachments,
            'headers': self.headers,
            'suspicious_headers': self.suspicious_headers
        }

    def get_analysis_text(self) -> str:
        """Get formatted text for LangChain agent analysis"""
        text = f"""
EMAIL ANALYSIS DATA:
===================

Subject: {self.subject}
From: {self.sender}
Sender Domain: {self.sender_domain}
To: {self.recipient}
Date: {self.date}

SUSPICIOUS HEADERS:
{self._format_dict(self.suspicious_headers)}

EXTRACTED URLs:
{self._format_list(self.urls)}

ATTACHMENTS:
{self._format_list(self.attachments)}

EMAIL BODY (First 1000 chars):
{self.body_text[:1000] if self.body_text else 'No text body'}

KEY HEADERS FOR ANALYSIS:
{self._format_dict(self.headers)}
"""
        return text

    def _format_dict(self, d: Dict) -> str:
        if not d:
            return "None"
        return "\n".join([f"  - {k}: {v}" for k, v in d.items()])

    def _format_list(self, lst: List) -> str:
        if not lst:
            return "None"
        return "\n".join([f"  - {item}" for item in lst])


class EmailParser:
    """Parse EML files and extract phishing-relevant features"""

    SUSPICIOUS_HEADER_KEYS = [
        'X-Sender-IP',
        'X-Originating-IP',
        'Authentication-Results',
        'Received-SPF',
        'X-Microsoft-Antispam',
        'X-Spam-Score',
        'X-SID-Result',
        'Return-Path'
    ]

    def __init__(self):
        self.url_pattern = re.compile(
            r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
        )

    def parse_eml_file(self, file_path: str) -> EmailData:
        """Parse an EML file and extract all relevant data"""
        with open(file_path, 'rb') as f:
            msg = BytesParser(policy=policy.default).parse(f)

        # Extract basic headers
        subject = msg.get('Subject', '')
        sender = msg.get('From', '')
        recipient = msg.get('To', '')
        date = msg.get('Date', '')

        # Extract sender domain
        sender_domain = self._extract_domain(sender)

        # Extract body
        body_text, body_html = self._extract_body(msg)

        # Extract URLs from body
        urls = self._extract_urls(body_text + body_html)

        # Extract attachments
        attachments = self._extract_attachments(msg)

        # Extract key headers
        headers = self._extract_key_headers(msg)

        # Extract suspicious headers
        suspicious_headers = self._extract_suspicious_headers(msg)

        return EmailData(
            file_path=file_path,
            subject=subject,
            sender=sender,
            sender_domain=sender_domain,
            recipient=recipient,
            date=date,
            body_text=body_text,
            body_html=body_html,
            urls=urls,
            attachments=attachments,
            headers=headers,
            suspicious_headers=suspicious_headers
        )

    def _extract_domain(self, email_address: str) -> str:
        """Extract domain from email address"""
        match = re.search(r'@([a-zA-Z0-9.-]+)', email_address)
        return match.group(1) if match else ''

    def _extract_body(self, msg) -> tuple:
        """Extract text and HTML body from email"""
        body_text = ''
        body_html = ''

        if msg.is_multipart():
            for part in msg.walk():
                content_type = part.get_content_type()
                content_disposition = str(part.get('Content-Disposition', ''))

                # Skip attachments
                if 'attachment' in content_disposition:
                    continue

                try:
                    payload = part.get_payload(decode=True)
                    if payload:
                        decoded = payload.decode('utf-8', errors='ignore')
                        if content_type == 'text/plain':
                            body_text += decoded
                        elif content_type == 'text/html':
                            body_html += decoded
                except:
                    pass
        else:
            try:
                payload = msg.get_payload(decode=True)
                if payload:
                    decoded = payload.decode('utf-8', errors='ignore')
                    content_type = msg.get_content_type()
                    if content_type == 'text/plain':
                        body_text = decoded
                    elif content_type == 'text/html':
                        body_html = decoded
            except:
                pass

        # If we only have HTML, extract some text from it
        if not body_text and body_html:
            body_text = self._html_to_text(body_html)

        return body_text.strip(), body_html.strip()

    def _html_to_text(self, html: str) -> str:
        """Simple HTML to text conversion"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', ' ', html)
        # Decode HTML entities
        text = unescape(text)
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _extract_urls(self, text: str) -> List[str]:
        """Extract all URLs from text"""
        urls = self.url_pattern.findall(text)
        # Remove duplicates while preserving order
        seen = set()
        unique_urls = []
        for url in urls:
            if url not in seen:
                seen.add(url)
                unique_urls.append(url)
        return unique_urls

    def _extract_attachments(self, msg) -> List[str]:
        """Extract attachment filenames"""
        attachments = []
        for part in msg.walk():
            if part.get_content_disposition() == 'attachment':
                filename = part.get_filename()
                if filename:
                    attachments.append(filename)
        return attachments

    def _extract_key_headers(self, msg) -> Dict[str, str]:
        """Extract important headers for analysis"""
        key_headers = {}
        important_keys = [
            'Subject', 'From', 'To', 'Date', 'Message-ID',
            'Content-Type', 'MIME-Version', 'Reply-To'
        ]
        for key in important_keys:
            value = msg.get(key)
            if value:
                key_headers[key] = str(value)
        return key_headers

    def _extract_suspicious_headers(self, msg) -> Dict[str, str]:
        """Extract headers that might indicate phishing"""
        suspicious = {}
        for key in self.SUSPICIOUS_HEADER_KEYS:
            value = msg.get(key)
            if value:
                suspicious[key] = str(value)
        return suspicious

    def parse_directory(self, directory: str, limit: Optional[int] = None) -> List[EmailData]:
        """Parse all EML files in a directory"""
        email_dir = Path(directory)
        eml_files = sorted(email_dir.glob('*.eml'))

        if limit:
            eml_files = eml_files[:limit]

        emails = []
        for eml_file in eml_files:
            try:
                email_data = self.parse_eml_file(str(eml_file))
                emails.append(email_data)
            except Exception as e:
                print(f"Error parsing {eml_file}: {e}")

        return emails


def main():
    """Example usage"""
    parser = EmailParser()

    # Parse a single file
    print("Parsing sample email...")
    email_data = parser.parse_eml_file('email/sample-1.eml')

    print("\n" + "="*80)
    print("PARSED EMAIL DATA:")
    print("="*80)
    print(email_data.get_analysis_text())

    # Parse multiple files
    print("\n" + "="*80)
    print("Parsing first 5 emails...")
    print("="*80)
    emails = parser.parse_directory('email', limit=5)
    print(f"\nSuccessfully parsed {len(emails)} emails")

    for i, email in enumerate(emails, 1):
        print(f"\n{i}. {email.subject[:60]}...")
        print(f"   From: {email.sender}")
        print(f"   URLs found: {len(email.urls)}")
        print(f"   Suspicious headers: {len(email.suspicious_headers)}")


if __name__ == '__main__':
    main()
