#!/usr/bin/env python3
"""
Build a ChromaDB vector store from email files.

Usage:
    python build_vector_store.py
"""

import os
from pathlib import Path
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_loader import PhishingEmailLoader
from tqdm import tqdm


def build_vector_store(
    email_directory: str = "smaller_email",
    persist_directory: str = "./chroma_db",
    chunk_size: int = 1000,
    chunk_overlap: int = 200
):
    """
    Build a ChromaDB vector store from emails.

    Args:
        email_directory: Directory containing .eml files
        persist_directory: Where to save the ChromaDB database
        chunk_size: Size of text chunks for embedding
        chunk_overlap: Overlap between chunks
    """

    print("="*80)
    print("BUILDING EMAIL VECTOR STORE WITH CHROMADB")
    print("="*80)

    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ Error: OPENAI_API_KEY not set!")
        print("Please set your API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return None

    print(f"\n✓ OpenAI API key found")
    print(f"✓ Email directory: {email_directory}")
    print(f"✓ Persist directory: {persist_directory}")
    print(f"✓ Chunk size: {chunk_size}, Overlap: {chunk_overlap}")

    # Load emails
    print(f"\n[1/4] Loading emails from {email_directory}...")
    loader = PhishingEmailLoader(email_directory)
    documents = loader.load()
    print(f"✓ Loaded {len(documents)} emails")

    # Split into chunks
    print(f"\n[2/4] Splitting documents into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_documents(documents)
    print(f"✓ Created {len(chunks)} chunks")

    # Create embeddings
    print(f"\n[3/4] Creating embeddings (this may take a minute)...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Build vector store
    print(f"\n[4/4] Building ChromaDB vector store...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name="phishing_emails"
    )

    print(f"\n✅ Vector store created successfully!")
    print(f"✓ Total documents: {len(documents)}")
    print(f"✓ Total chunks: {len(chunks)}")
    print(f"✓ Stored in: {persist_directory}")
    print(f"\n{'='*80}")

    return vectorstore


def get_collection_stats(persist_directory: str = "./chroma_db"):
    """Get statistics about the vector store"""
    if not Path(persist_directory).exists():
        print(f"❌ Vector store not found at {persist_directory}")
        return

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings,
        collection_name="phishing_emails"
    )

    # Get collection
    collection = vectorstore._collection
    count = collection.count()

    print(f"\n{'='*80}")
    print("VECTOR STORE STATISTICS")
    print(f"{'='*80}")
    print(f"Total vectors: {count}")
    print(f"Location: {persist_directory}")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Build email vector store with ChromaDB')
    parser.add_argument('--email-dir', default='smaller_email', help='Email directory')
    parser.add_argument('--db-dir', default='./chroma_db', help='ChromaDB persist directory')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size')
    parser.add_argument('--chunk-overlap', type=int, default=200, help='Chunk overlap')
    parser.add_argument('--stats', action='store_true', help='Show stats only')

    args = parser.parse_args()

    if args.stats:
        get_collection_stats(args.db_dir)
    else:
        vectorstore = build_vector_store(
            email_directory=args.email_dir,
            persist_directory=args.db_dir,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap
        )

        if vectorstore:
            print("\nVector store is ready for similarity search!")
            print("Run: python search_similar_emails.py --help")
