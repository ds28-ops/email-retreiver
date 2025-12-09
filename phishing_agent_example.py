"""
Example LangChain agent for phishing email detection.
This demonstrates how to use the prepared email data with a LangChain agent.
"""

from typing import List, Dict, Any
from langchain_loader import PhishingEmailLoader, PhishingEmailVectorStorePrep
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# Phishing detection prompt template
PHISHING_ANALYSIS_PROMPT = """You are a cybersecurity expert specializing in phishing email detection.
Analyze the following email and determine if it is a phishing attempt.

Look for these phishing indicators:
1. Suspicious sender domain (doesn't match claimed organization)
2. Urgent or threatening language
3. Requests for personal/financial information
4. Suspicious URLs or mismatched links
5. Poor grammar or spelling
6. Generic greetings
7. Unexpected attachments
8. SPF/DKIM authentication failures
9. Spoofed sender addresses
10. Too-good-to-be-true offers

Email to analyze:
{email_content}

Provide a detailed analysis with:
1. **Phishing Risk Score** (0-100): Your confidence this is phishing
2. **Risk Level**: LOW, MEDIUM, HIGH, or CRITICAL
3. **Indicators Found**: List all phishing indicators detected
4. **Explanation**: Detailed reasoning for your assessment
5. **Recommended Action**: What the recipient should do

Format your response clearly with these sections.
"""


class PhishingDetectionAgent:
    """Agent for detecting phishing emails using LangChain"""

    def __init__(self, llm=None):
        """
        Initialize the phishing detection agent.

        Args:
            llm: LangChain LLM instance (e.g., ChatAnthropic, ChatOpenAI)
                 If None, examples will show structure only
        """
        self.llm = llm
        self.email_loader = None

        if llm:
            self.chain = (
                ChatPromptTemplate.from_template(PHISHING_ANALYSIS_PROMPT)
                | llm
                | StrOutputParser()
            )

    def load_emails(self, email_directory: str, limit: int = None):
        """Load emails from directory"""
        self.email_loader = PhishingEmailLoader(email_directory, limit=limit)
        return self.email_loader.load()

    def analyze_email(self, email_content: str) -> str:
        """
        Analyze a single email for phishing indicators.

        Args:
            email_content: The email content to analyze

        Returns:
            Analysis result as string
        """
        if not self.llm:
            return "LLM not configured. Please provide an LLM instance."

        return self.chain.invoke({"email_content": email_content})

    def analyze_document(self, document) -> Dict[str, Any]:
        """
        Analyze a LangChain document (from PhishingEmailLoader).

        Args:
            document: LangChain Document object

        Returns:
            Dictionary with analysis results
        """
        analysis = self.analyze_email(document.page_content)

        return {
            'file': document.metadata.get('source'),
            'subject': document.metadata.get('subject'),
            'sender': document.metadata.get('sender'),
            'analysis': analysis,
            'metadata': document.metadata
        }

    def batch_analyze(self, documents: List, verbose: bool = True) -> List[Dict]:
        """
        Analyze multiple emails in batch.

        Args:
            documents: List of LangChain Document objects
            verbose: Whether to print progress

        Returns:
            List of analysis results
        """
        results = []
        total = len(documents)

        for i, doc in enumerate(documents, 1):
            if verbose:
                print(f"Analyzing email {i}/{total}: {doc.metadata.get('subject', 'No subject')[:50]}...")

            result = self.analyze_document(doc)
            results.append(result)

        return results


def example_without_llm():
    """Example showing data preparation without LLM"""
    print("="*80)
    print("PHISHING DETECTION - DATA PREPARATION EXAMPLE")
    print("="*80)

    # Load and prepare emails
    print("\nLoading emails from 'email' directory...")
    loader = PhishingEmailLoader('email', limit=3)
    documents = loader.load()

    print(f"\nLoaded {len(documents)} emails for analysis")

    # Display what data is available for the agent
    for i, doc in enumerate(documents, 1):
        print(f"\n{'-'*80}")
        print(f"EMAIL {i}")
        print(f"{'-'*80}")
        print(f"Subject: {doc.metadata['subject']}")
        print(f"Sender: {doc.metadata['sender']}")
        print(f"Domain: {doc.metadata['sender_domain']}")
        print(f"URLs found: {doc.metadata['url_count']}")
        if doc.metadata['urls']:
            print(f"First URL: {doc.metadata['urls'][0]}")
        print(f"Suspicious headers: {doc.metadata['suspicious_header_count']}")
        print(f"\nContent preview (first 500 chars):")
        print(doc.page_content[:500])
        print("...")

    # Show vector store preparation
    print("\n" + "="*80)
    print("VECTOR STORE PREPARATION")
    print("="*80)
    prep = PhishingEmailVectorStorePrep('email')
    vector_docs = prep.prepare_documents(limit=5, chunk_size=1000)
    print(f"\nPrepared {len(vector_docs)} chunks for vector store")


def example_with_llm():
    """Example with actual LLM (requires API key)"""
    print("="*80)
    print("PHISHING DETECTION WITH LLM")
    print("="*80)

    # Uncomment and configure your LLM of choice:
    #
    # Option 1: Claude (Anthropic)
    # from langchain_anthropic import ChatAnthropic
    # llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
    #
    # Option 2: OpenAI
    # from langchain_openai import ChatOpenAI
    # llm = ChatOpenAI(model="gpt-4")
    #
    # Then:
    # agent = PhishingDetectionAgent(llm=llm)
    # documents = agent.load_emails('email', limit=2)
    # results = agent.batch_analyze(documents)
    #
    # for result in results:
    #     print(f"\n{'='*80}")
    #     print(f"Analysis for: {result['subject']}")
    #     print(f"{'='*80}")
    #     print(result['analysis'])

    print("\nTo use LLM analysis:")
    print("1. Install: pip install langchain-anthropic  (or langchain-openai)")
    print("2. Set API key: export ANTHROPIC_API_KEY='your-key'  (or OPENAI_API_KEY)")
    print("3. Uncomment the code above in this function")
    print("\nFor now, showing data preparation only...")

    example_without_llm()


def create_retrieval_chain_example():
    """Example of using emails with a retrieval chain"""
    print("\n" + "="*80)
    print("RETRIEVAL CHAIN SETUP EXAMPLE")
    print("="*80)

    print("""
To create a retrieval-based phishing detector:

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain_anthropic import ChatAnthropic

# 1. Load and prepare emails
prep = PhishingEmailVectorStorePrep('email')
documents = prep.prepare_documents(limit=100)

# 2. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(documents, embeddings)

# 3. Create retrieval chain
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022")
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3})
)

# 4. Query similar phishing patterns
result = qa_chain.invoke({
    "query": "Find emails with suspicious banking requests"
})
```
""")


if __name__ == '__main__':
    print("\nRUNNING PHISHING DETECTION EXAMPLES\n")

    # Run data preparation example
    example_without_llm()

    # Show retrieval chain setup
    create_retrieval_chain_example()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("""
You now have:
1. ✓ Email parser (email_parser.py) - Extracts phishing indicators
2. ✓ LangChain loader (langchain_loader.py) - Loads emails as Documents
3. ✓ Agent example (this file) - Shows how to analyze emails

Next steps:
- Install LangChain: pip install langchain langchain-anthropic
- Set API key: export ANTHROPIC_API_KEY='your-key'
- Run analysis on your phishing samples
- Build a vector store for similarity search
- Create custom phishing detection tools
""")
