#!/usr/bin/env python3
"""
Quick start script for phishing email analysis with LangChain.

Usage:
    python run_analysis.py --limit 10
    python run_analysis.py --all
    python run_analysis.py --file email/sample-1.eml
"""

import argparse
import sys
from pathlib import Path
from email_parser import EmailParser
from langchain_loader import PhishingEmailLoader, PhishingEmailVectorStorePrep


def analyze_single_email(file_path: str):
    """Analyze a single email file"""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {file_path}")
    print(f"{'='*80}\n")

    parser = EmailParser()
    try:
        email_data = parser.parse_eml_file(file_path)
        print(email_data.get_analysis_text())

        # Print phishing risk indicators
        print("\n" + "="*80)
        print("PHISHING RISK INDICATORS:")
        print("="*80)

        risk_score = 0
        indicators = []

        # Check for SPF/DKIM issues
        if 'spf=fail' in str(email_data.suspicious_headers).lower():
            indicators.append("❌ SPF authentication failed")
            risk_score += 20
        elif 'spf=temperror' in str(email_data.suspicious_headers).lower():
            indicators.append("⚠️  SPF temporary error")
            risk_score += 10

        if 'dkim=fail' in str(email_data.suspicious_headers).lower():
            indicators.append("❌ DKIM authentication failed")
            risk_score += 20
        elif 'dkim=none' in str(email_data.suspicious_headers).lower():
            indicators.append("⚠️  No DKIM signature")
            risk_score += 10

        # Check for domain mismatch
        if email_data.sender and email_data.sender_domain:
            claimed_org = email_data.sender.lower()
            if any(brand in claimed_org for brand in ['microsoft', 'google', 'amazon', 'apple', 'paypal', 'bank']):
                if not any(brand in email_data.sender_domain.lower() for brand in ['microsoft', 'google', 'amazon', 'apple', 'paypal']):
                    indicators.append(f"❌ Domain mismatch: Claims to be from major brand but uses '{email_data.sender_domain}'")
                    risk_score += 30

        # Check for suspicious URLs
        if email_data.urls:
            suspicious_patterns = ['bit.ly', 'tinyurl', '.tk', '.ml', '.ga', 'track', 'click']
            for url in email_data.urls:
                if any(pattern in url.lower() for pattern in suspicious_patterns):
                    indicators.append(f"⚠️  Suspicious URL pattern: {url[:80]}...")
                    risk_score += 15
                    break

        # Check for urgency in subject
        urgent_words = ['urgent', 'expir', 'suspend', 'verify', 'confirm', 'limited time', 'act now']
        if any(word in email_data.subject.lower() for word in urgent_words):
            indicators.append(f"⚠️  Urgent language in subject")
            risk_score += 10

        # Check for unusual sender IP
        if 'X-Sender-IP' in email_data.suspicious_headers:
            indicators.append(f"ℹ️  Sender IP: {email_data.suspicious_headers['X-Sender-IP']}")

        # Display results
        if not indicators:
            indicators.append("✅ No obvious phishing indicators detected")

        for indicator in indicators:
            print(indicator)

        print(f"\n{'='*80}")
        print(f"ESTIMATED RISK SCORE: {min(risk_score, 100)}/100")
        if risk_score >= 60:
            print("RISK LEVEL: 🔴 CRITICAL - Likely phishing")
        elif risk_score >= 40:
            print("RISK LEVEL: 🟠 HIGH - Suspicious")
        elif risk_score >= 20:
            print("RISK LEVEL: 🟡 MEDIUM - Use caution")
        else:
            print("RISK LEVEL: 🟢 LOW - Appears legitimate")
        print(f"{'='*80}\n")

    except Exception as e:
        print(f"Error analyzing {file_path}: {e}")
        import traceback
        traceback.print_exc()


def analyze_batch(limit=None):
    """Analyze multiple emails"""
    print(f"\n{'='*80}")
    print(f"BATCH ANALYSIS - Loading emails...")
    print(f"{'='*80}\n")

    loader = PhishingEmailLoader('email', limit=limit)
    documents = loader.load()

    print(f"Loaded {len(documents)} emails\n")

    # Summary statistics
    total_urls = 0
    total_suspicious_headers = 0
    spf_issues = 0
    dkim_issues = 0

    for doc in documents:
        total_urls += doc.metadata['url_count']
        total_suspicious_headers += doc.metadata['suspicious_header_count']
        if doc.metadata.get('has_spf_issues'):
            spf_issues += 1
        if doc.metadata.get('has_dkim_issues'):
            dkim_issues += 1

    print(f"{'='*80}")
    print("SUMMARY STATISTICS:")
    print(f"{'='*80}")
    print(f"Total emails analyzed: {len(documents)}")
    print(f"Total URLs extracted: {total_urls}")
    print(f"Average URLs per email: {total_urls/len(documents):.1f}")
    print(f"Emails with SPF issues: {spf_issues} ({spf_issues/len(documents)*100:.1f}%)")
    print(f"Emails with DKIM issues: {dkim_issues} ({dkim_issues/len(documents)*100:.1f}%)")
    print(f"Average suspicious headers: {total_suspicious_headers/len(documents):.1f}")
    print(f"{'='*80}\n")

    # Show top suspicious emails
    print("TOP 5 SUSPICIOUS EMAILS:")
    print(f"{'='*80}")
    sorted_docs = sorted(documents, key=lambda d: d.metadata['url_count'], reverse=True)
    for i, doc in enumerate(sorted_docs[:5], 1):
        print(f"{i}. {doc.metadata['subject'][:70]}")
        print(f"   From: {doc.metadata['sender'][:60]}")
        print(f"   URLs: {doc.metadata['url_count']}, Suspicious headers: {doc.metadata['suspicious_header_count']}")
        print()


def prepare_for_vector_store(limit=None, output_file='emails_for_vectorstore.txt'):
    """Prepare emails for vector store ingestion"""
    print(f"\n{'='*80}")
    print(f"PREPARING FOR VECTOR STORE")
    print(f"{'='*80}\n")

    prep = PhishingEmailVectorStorePrep('email')
    documents = prep.prepare_documents(limit=limit, chunk_size=1000)

    print(f"Prepared {len(documents)} document chunks")
    print(f"Writing to {output_file}...")

    with open(output_file, 'w') as f:
        for i, doc in enumerate(documents, 1):
            f.write(f"\n{'='*80}\n")
            f.write(f"CHUNK {i}\n")
            f.write(f"{'='*80}\n")
            f.write(f"Source: {doc.metadata.get('source', 'unknown')}\n")
            f.write(f"Subject: {doc.metadata.get('subject', 'N/A')}\n")
            f.write(f"\nContent:\n{doc.page_content}\n")

    print(f"✅ Saved to {output_file}")
    print(f"\nYou can now use these documents with embeddings:")
    print("""
    from langchain_openai import OpenAIEmbeddings
    from langchain_community.vectorstores import FAISS

    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(documents, embeddings)
    """)


def main():
    parser = argparse.ArgumentParser(
        description='Analyze phishing emails for LangChain agent',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_analysis.py --limit 10              # Analyze first 10 emails
  python run_analysis.py --all                   # Analyze all emails
  python run_analysis.py --file email/sample-1.eml  # Analyze single email
  python run_analysis.py --prepare-vector        # Prepare for vector store
        """
    )

    parser.add_argument('--file', help='Analyze a specific email file')
    parser.add_argument('--limit', type=int, help='Number of emails to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all emails')
    parser.add_argument('--prepare-vector', action='store_true', help='Prepare emails for vector store')
    parser.add_argument('--output', default='emails_for_vectorstore.txt', help='Output file for vector store prep')

    args = parser.parse_args()

    # Check if email directory exists
    if not Path('email').exists():
        print("Error: 'email' directory not found!")
        print("Make sure you run this script from the phishing_pot directory.")
        sys.exit(1)

    # Execute based on arguments
    if args.file:
        analyze_single_email(args.file)
    elif args.prepare_vector:
        prepare_for_vector_store(args.limit, args.output)
    elif args.all:
        analyze_batch(limit=None)
    elif args.limit:
        analyze_batch(limit=args.limit)
    else:
        # Default: analyze first 5 emails
        print("No arguments specified. Analyzing first 5 emails...")
        print("Use --help for more options.\n")
        analyze_batch(limit=5)


if __name__ == '__main__':
    main()
