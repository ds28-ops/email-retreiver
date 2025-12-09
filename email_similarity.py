#!/usr/bin/env python3
"""
Build ChromaDB vector store and search for similar emails using LangChain.

Usage:
    # Step 1: Build the vector store
    python email_similarity.py --build

    # Step 2: Search for similar emails
    python email_similarity.py --search smaller_email/sample-1.eml --top 5
"""

import os
import argparse
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_loader import PhishingEmailLoader
from email_parser import EmailParser


# Your OpenAI API key
API_KEY = ''

# Set it in environment
os.environ['OPENAI_API_KEY'] = API_KEY


def build_vector_store(email_directory="smaller_email", persist_directory="./chroma_db"):
    """Build ChromaDB vector store from emails"""
    print("="*80)
    print("STEP 1: BUILDING VECTOR STORE")
    print("="*80)

    # Load emails using LangChain loader
    print(f"\n[1/4] Loading emails from {email_directory}...")
    loader = PhishingEmailLoader(email_directory)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} emails")

    # Split into chunks with RecursiveCharacterTextSplitter
    print(f"\n[2/4] Splitting into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")

    # Create OpenAI embeddings
    print(f"\n[3/4] Creating OpenAI embeddings...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY
    )
    print(f"✓ Embeddings initialized")

    # Build ChromaDB vector store
    print(f"\n[4/4] Building ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="phishing_emails"
    )

    print(f"\n{'='*80}")
    print(f"✅ VECTOR STORE BUILT SUCCESSFULLY!")
    print(f"{'='*80}")
    print(f"Total emails: {len(documents)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Location: {persist_directory}")
    print(f"{'='*80}\n")

    return vectorstore


def search_similar_emails(email_file, top_k=5, persist_directory="./chroma_db"):
    """Search for similar emails using similarity metric"""
    print("="*80)
    print("STEP 2: SEARCHING FOR SIMILAR EMAILS")
    print("="*80)

    # Check if vector store exists
    if not Path(persist_directory).exists():
        print(f"\n❌ Vector store not found at {persist_directory}")
        print(f"Please run: python email_similarity.py --build")
        return

    # Load the vector store
    print(f"\n[1/3] Loading vector store from {persist_directory}...")
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=API_KEY
    )
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="phishing_emails"
    )
    print(f"✓ Vector store loaded")

    # Parse the query email
    print(f"\n[2/3] Parsing query email: {email_file}")
    parser = EmailParser()
    email_data = parser.parse_eml_file(email_file)

    print(f"\nQuery Email:")
    print(f"  Subject: {email_data.subject[:80]}")
    print(f"  From: {email_data.sender}")
    print(f"  URLs: {len(email_data.urls)}")

    # Use email content as query
    query_text = email_data.get_analysis_text()

    # Search with similarity metric
    print(f"\n[3/3] Searching for top {top_k} similar emails...")
    results = vectorstore.similarity_search_with_score(query_text, k=top_k)

    # Display results - ONLY EMAIL BODY
    print(f"\n{'='*80}")
    print(f"TOP {len(results)} SIMILAR EMAILS - EMAIL BODY ONLY")
    print(f"{'='*80}\n")

    # Group by source file to get unique emails
    seen_sources = {}
    for doc, score in results:
        source = doc.metadata.get('source', 'unknown')
        if source not in seen_sources:
            seen_sources[source] = (doc, score)

    # Parse and display email bodies
    parser = EmailParser()
    for i, (source, (doc, score)) in enumerate(list(seen_sources.items())[:top_k], 1):
        # Parse the original email to get the full body
        try:
            email_data = parser.parse_eml_file(source)
            email_body = email_data.body_text
        except:
            email_body = "Could not parse email body"

        print(f"\n{'='*80}")
        print(f"EMAIL #{i}")
        print(f"{'='*80}")
        print(f"Similarity: {1-score:.4f}")
        print(f"Source: {source}")
        print(f"Subject: {doc.metadata.get('subject', 'N/A')[:80]}")
        print(f"From: {doc.metadata.get('sender', 'N/A')}")
        print(f"\nEMAIL BODY:")
        print(f"{'-'*80}")
        print(email_body)
        print(f"{'='*80}\n")

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Email similarity search with ChromaDB and OpenAI embeddings',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build the vector store
  python email_similarity.py --build

  # Search for similar emails
  python email_similarity.py --search smaller_email/sample-1.eml --top 5
        """
    )

    parser.add_argument('--build', action='store_true', help='Build the vector store')
    parser.add_argument('--search', help='Email file to search for similar emails')
    parser.add_argument('--top', type=int, default=5, help='Number of similar emails (default: 5)')
    parser.add_argument('--email-dir', default='smaller_email', help='Email directory')
    parser.add_argument('--db-dir', default='./chroma_db', help='ChromaDB directory')

    args = parser.parse_args()

    if args.build:
        build_vector_store(args.email_dir, args.db_dir)
    elif args.search:
        search_similar_emails(args.search, args.top, args.db_dir)
    else:
        print("Please specify --build or --search")
        parser.print_help()


if __name__ == '__main__':
    main()
