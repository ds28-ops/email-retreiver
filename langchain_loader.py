from typing import List, Iterator, Optional, Dict
from pathlib import Path
from langchain_core.documents import Document
from langchain_core.document_loaders import BaseLoader
from email_parser import EmailParser, EmailData


class PhishingEmailLoader(BaseLoader):
    """LangChain document loader for phishing email analysis"""

    def __init__(
        self,
        email_directory: str,
        limit: Optional[int] = None,
        include_html: bool = False
    ):
        """
        Initialize the phishing email loader.

        Args:
            email_directory: Path to directory containing .eml files
            limit: Maximum number of emails to load (None for all)
            include_html: Whether to include HTML body in metadata
        """
        self.email_directory = email_directory
        self.limit = limit
        self.include_html = include_html
        self.parser = EmailParser()

    def lazy_load(self) -> Iterator[Document]:
        """Lazy load emails one at a time"""
        email_dir = Path(self.email_directory)
        eml_files = sorted(email_dir.glob('*.eml'))

        if self.limit:
            eml_files = eml_files[:self.limit]

        for eml_file in eml_files:
            try:
                email_data = self.parser.parse_eml_file(str(eml_file))
                yield self._create_document(email_data)
            except Exception as e:
                print(f"Error loading {eml_file}: {e}")

    def load(self) -> List[Document]:
        """Load all emails into memory"""
        return list(self.lazy_load())

    def _create_document(self, email_data: EmailData) -> Document:
        """Convert EmailData to LangChain Document"""
        # Create the main content that the LLM will analyze
        page_content = email_data.get_analysis_text()

        # Create metadata for filtering and additional context
        # Note: ChromaDB only supports str, int, float, bool, None - no lists!
        metadata = {
            'source': email_data.file_path,
            'subject': email_data.subject,
            'sender': email_data.sender,
            'sender_domain': email_data.sender_domain,
            'recipient': email_data.recipient,
            'date': email_data.date,
            'url_count': len(email_data.urls),
            'urls': ', '.join(email_data.urls) if email_data.urls else '',  # Convert list to string
            'attachment_count': len(email_data.attachments),
            'attachments': ', '.join(email_data.attachments) if email_data.attachments else '',  # Convert list to string
            'suspicious_header_count': len(email_data.suspicious_headers),
            'has_spf_issues': 'SPF' in str(email_data.suspicious_headers),
            'has_dkim_issues': 'dkim' in str(email_data.suspicious_headers).lower(),
        }

        # Optionally include HTML in metadata
        if self.include_html and email_data.body_html:
            metadata['body_html'] = email_data.body_html[:5000]  # Limit HTML size

        return Document(
            page_content=page_content,
            metadata=metadata
        )


class PhishingEmailVectorStorePrep:
    """Prepare phishing emails for vector store ingestion"""

    def __init__(self, email_directory: str):
        self.email_directory = email_directory
        self.loader = PhishingEmailLoader(email_directory)

    def prepare_documents(
        self,
        limit: Optional[int] = None,
        chunk_size: Optional[int] = None
    ) -> List[Document]:
        """
        Prepare documents for vector store.

        Args:
            limit: Maximum number of emails to process
            chunk_size: If provided, split large emails into chunks

        Returns:
            List of LangChain Documents ready for embedding
        """
        self.loader.limit = limit
        documents = self.loader.load()

        if chunk_size:
            from langchain.text_splitter import RecursiveCharacterTextSplitter
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            documents = text_splitter.split_documents(documents)

        return documents

    def prepare_for_agent(self, limit: Optional[int] = None) -> List[Dict]:
        """
        Prepare emails as structured data for agent tool use.

        Returns:
            List of dictionaries with email analysis data
        """
        parser = EmailParser()
        emails = parser.parse_directory(self.email_directory, limit=limit)
        return [email.to_dict() for email in emails]


def main():
    """Example usage with LangChain"""
    print("="*80)
    print("LANGCHAIN PHISHING EMAIL LOADER - EXAMPLES")
    print("="*80)

    # Example 1: Load emails as LangChain documents
    print("\n1. Loading emails as LangChain Documents...")
    loader = PhishingEmailLoader('email', limit=3)
    documents = loader.load()

    print(f"\nLoaded {len(documents)} documents")
    for i, doc in enumerate(documents, 1):
        print(f"\nDocument {i}:")
        print(f"  Subject: {doc.metadata['subject'][:60]}...")
        print(f"  Sender: {doc.metadata['sender']}")
        print(f"  URLs: {doc.metadata['url_count']}")
        print(f"  Suspicious headers: {doc.metadata['suspicious_header_count']}")
        print(f"  Content length: {len(doc.page_content)} chars")

    # Example 2: Prepare for vector store
    print("\n" + "="*80)
    print("2. Preparing for Vector Store...")
    prep = PhishingEmailVectorStorePrep('email')
    vector_docs = prep.prepare_documents(limit=5)
    print(f"\nPrepared {len(vector_docs)} documents for vector store")

    # Example 3: Prepare structured data for agent
    print("\n" + "="*80)
    print("3. Preparing structured data for agent...")
    agent_data = prep.prepare_for_agent(limit=2)
    print(f"\nPrepared {len(agent_data)} emails as structured data")

    if agent_data:
        print("\nSample structured data:")
        sample = agent_data[0]
        print(f"  Subject: {sample['subject']}")
        print(f"  URLs: {len(sample['urls'])}")
        print(f"  First URL: {sample['urls'][0] if sample['urls'] else 'None'}")


if __name__ == '__main__':
    main()
