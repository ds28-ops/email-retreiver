#!/usr/bin/env python3
"""
Test phishing detection with OpenAI on 5 sample emails.

Usage:
    export OPENAI_API_KEY='your-key-here'
    python test_with_openai.py
"""

import os
from langchain_openai import ChatOpenAI
from phishing_agent_example import PhishingDetectionAgent, PHISHING_ANALYSIS_PROMPT
from langchain_loader import PhishingEmailLoader


def main():
    print("="*80)
    print("PHISHING DETECTION WITH OPENAI")
    print("="*80)

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("\n❌ Error: OPENAI_API_KEY not set!")
        print("Please set your API key:")
        print("  export OPENAI_API_KEY='your-key-here'")
        return

    print("\n✓ OpenAI API key found")

    # Initialize LLM and agent
    print("✓ Initializing OpenAI GPT-4...")
    llm = ChatOpenAI(
        model="gpt-4o-mini",  # Using mini for faster/cheaper testing
        temperature=0  # Deterministic output for security analysis
    )

    agent = PhishingDetectionAgent(llm=llm)

    # Load 5 emails
    print("✓ Loading 5 sample emails...\n")
    documents = agent.load_emails('email', limit=5)

    print(f"Loaded {len(documents)} emails")
    print("="*80)

    # Analyze each email
    results = []
    for i, doc in enumerate(documents, 1):
        print(f"\n{'='*80}")
        print(f"ANALYZING EMAIL {i}/{len(documents)}")
        print(f"{'='*80}")
        print(f"Subject: {doc.metadata['subject']}")
        print(f"From: {doc.metadata['sender']}")
        print(f"URLs found: {doc.metadata['url_count']}")
        print(f"\nAnalyzing with GPT-4...")
        print("-"*80)

        try:
            result = agent.analyze_document(doc)
            results.append(result)

            # Print the analysis
            print(result['analysis'])
            print("-"*80)

        except Exception as e:
            print(f"❌ Error analyzing email: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nSuccessfully analyzed {len(results)}/{len(documents)} emails")

    # Save results to file
    output_file = 'phishing_analysis_results.txt'
    print(f"\nSaving detailed results to {output_file}...")

    with open(output_file, 'w') as f:
        f.write("PHISHING EMAIL ANALYSIS RESULTS\n")
        f.write("="*80 + "\n\n")

        for i, result in enumerate(results, 1):
            f.write(f"\nEMAIL {i}\n")
            f.write("="*80 + "\n")
            f.write(f"File: {result['file']}\n")
            f.write(f"Subject: {result['subject']}\n")
            f.write(f"From: {result['sender']}\n")
            f.write(f"\nANALYSIS:\n")
            f.write("-"*80 + "\n")
            f.write(result['analysis'])
            f.write("\n\n" + "="*80 + "\n")

    print(f"✅ Results saved to {output_file}")

    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    for i, result in enumerate(results, 1):
        subject = result['subject'][:60] + "..." if len(result['subject']) > 60 else result['subject']
        print(f"{i}. {subject}")


if __name__ == '__main__':
    main()


"""
1. Extract email body from eml files, recursive character split, with embeddings
2. astore in chroma db vector store
3. vector retreiver, with the meytrics as similarity, and get the top 5
"""