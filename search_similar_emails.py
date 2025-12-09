#!/usr/bin/env python3
"""
Search for similar emails in the ChromaDB vector store.

Usage:
    # Search by submitting an email file
    python search_similar_emails.py --email smaller_email/sample-1.eml

    # Search by query text
    python search_similar_emails.py --query "urgent password reset request"

    # Get top 10 results
    python search_similar_emails.py --email smaller_email/sample-1.eml --top 10
"""

import os
import argparse
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from email_parser import EmailParser


def load_vector_store(persist_directory: str = "./chroma_db"):
    """Load the ChromaDB vector store"""
    if not Path(persist_directory).exists():
        print(f"❌ Vector store not found at {persist_directory}")
        print(f"Please run: python build_vector_store.py")
        return None

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="phishing_emails"
    )

    return vectorstore


def search_similar_emails_by_file(
    email_file: str,
    vectorstore,
    top_k: int = 5,
    print_full_content: bool = False
):
    """
    Search for similar emails by providing an email file.

    Args:
        email_file: Path to .eml file
        vectorstore: ChromaDB vector store
        top_k: Number of similar emails to return
        print_full_content: Whether to print full email content
    """
    print("="*80)
    print(f"SEARCHING FOR SIMILAR EMAILS")
    print("="*80)

    # Parse the email
    print(f"\nParsing email: {email_file}")
    parser = EmailParser()
    email_data = parser.parse_eml_file(email_file)

    # Use the email's analysis text as query
    query_text = email_data.get_analysis_text()

    print(f"\nQuery email details:")
    print(f"  Subject: {email_data.subject[:80]}")
    print(f"  From: {email_data.sender}")
    print(f"  URLs: {len(email_data.urls)}")
    print(f"  Suspicious headers: {len(email_data.suspicious_headers)}")

    # Search for similar documents
    print(f"\nSearching for top {top_k} similar emails...")
    results = vectorstore.similarity_search_with_score(query_text, k=top_k)

    # Display results
    print(f"\n{'='*80}")
    print(f"FOUND {len(results)} SIMILAR EMAILS")
    print(f"{'='*80}\n")

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"RESULT #{i} - Similarity Score: {1-score:.4f} (distance: {score:.4f})")
        print(f"{'─'*80}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Subject: {doc.metadata.get('subject', 'N/A')}")
        print(f"From: {doc.metadata.get('sender', 'N/A')}")
        print(f"Domain: {doc.metadata.get('sender_domain', 'N/A')}")
        print(f"URLs: {doc.metadata.get('url_count', 0)}")
        print(f"Attachments: {doc.metadata.get('attachment_count', 0)}")

        if doc.metadata.get('urls'):
            print(f"\nURLs found:")
            for url in doc.metadata['urls'][:3]:  # Show first 3 URLs
                print(f"  - {url}")

        if print_full_content:
            print(f"\nFull content:")
            print(doc.page_content)
        else:
            print(f"\nContent preview (first 300 chars):")
            print(doc.page_content[:300] + "...")

    return results


def search_similar_emails_by_query(
    query: str,
    vectorstore,
    top_k: int = 5,
    print_full_content: bool = False
):
    """
    Search for similar emails by query text.

    Args:
        query: Search query
        vectorstore: ChromaDB vector store
        top_k: Number of similar emails to return
        print_full_content: Whether to print full email content
    """
    print("="*80)
    print(f"SEARCHING FOR SIMILAR EMAILS")
    print("="*80)

    print(f"\nQuery: {query}")

    # Search for similar documents
    print(f"\nSearching for top {top_k} similar emails...")
    results = vectorstore.similarity_search_with_score(query, k=top_k)

    # Display results
    print(f"\n{'='*80}")
    print(f"FOUND {len(results)} SIMILAR EMAILS")
    print(f"{'='*80}\n")

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{'─'*80}")
        print(f"RESULT #{i} - Similarity Score: {1-score:.4f} (distance: {score:.4f})")
        print(f"{'─'*80}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Subject: {doc.metadata.get('subject', 'N/A')}")
        print(f"From: {doc.metadata.get('sender', 'N/A')}")
        print(f"Domain: {doc.metadata.get('sender_domain', 'N/A')}")
        print(f"URLs: {doc.metadata.get('url_count', 0)}")

        if doc.metadata.get('urls'):
            print(f"\nURLs found:")
            for url in doc.metadata['urls'][:3]:  # Show first 3 URLs
                print(f"  - {url}")

        if print_full_content:
            print(f"\nFull content:")
            print(doc.page_content)
        else:
            print(f"\nContent preview (first 300 chars):")
            print(doc.page_content[:300] + "...")

    return results


def print_full_email_contents(results):
    """Print the full content of all result emails"""
    print("\n" + "="*80)
    print("FULL EMAIL CONTENTS")
    print("="*80)

    for i, (doc, score) in enumerate(results, 1):
        print(f"\n{'='*80}")
        print(f"EMAIL #{i}")
        print(f"{'='*80}")
        print(f"Source: {doc.metadata.get('source', 'unknown')}")
        print(f"Similarity Score: {1-score:.4f}")
        print(f"\n{doc.page_content}")
        print(f"\n{'='*80}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Search for similar emails in ChromaDB vector store',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Search by email file
  python search_similar_emails.py --email smaller_email/sample-1.eml

  # Search by query
  python search_similar_emails.py --query "password reset bank"

  # Get top 10 results with full content
  python search_similar_emails.py --email smaller_email/sample-1.eml --top 10 --full
        """
    )

    parser.add_argument('--email', help='Path to email file to find similar emails')
    parser.add_argument('--query', help='Search query text')
    parser.add_argument('--top', type=int, default=5, help='Number of results (default: 5)')
    parser.add_argument('--full', action='store_true', help='Print full email content')
    parser.add_argument('--db-dir', default='./chroma_db', help='ChromaDB directory')

    args = parser.parse_args()

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("❌ Error: OPENAI_API_KEY not set!")
        print("Please set your API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        exit(1)

    # Load vector store
    vectorstore = load_vector_store(args.db_dir)
    if not vectorstore:
        exit(1)

    # Perform search
    if args.email:
        results = search_similar_emails_by_file(
            args.email,
            vectorstore,
            top_k=args.top,
            print_full_content=args.full
        )
    elif args.query:
        results = search_similar_emails_by_query(
            args.query,
            vectorstore,
            top_k=args.top,
            print_full_content=args.full
        )
    else:
        print("❌ Error: Please provide either --email or --query")
        parser.print_help()
        exit(1)

    # Optionally print full contents
    if not args.full:
        print(f"\n{'='*80}")
        print("TIP: Use --full flag to see complete email contents")
        print(f"{'='*80}")
