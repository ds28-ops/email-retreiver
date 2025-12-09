# Email Similarity Search - Quick Start Guide
## Outputs can be found in the outputs folder.

## What This Does

This system uses LangChain, OpenAI embeddings, and ChromaDB to find similar phishing emails based on content similarity.

## Files Created

- **email_parser.py** - Parses EML files and extracts features
- **langchain_loader.py** - LangChain document loader for emails
- **email_similarity.py** - Main script for building vector store and searching
- **chroma_db/** - Local ChromaDB vector store (created after running --build)

## Quick Start

### Step 1: Build the Vector Store

This loads all 100 emails from `smaller_email/`, chunks them, creates embeddings, and stores in ChromaDB:

```bash
python email_similarity.py --build
```

**Output:**
- Loads 100 emails
- Creates ~538 chunks
- Stores embeddings in `./chroma_db/`
- Uses OpenAI text-embedding-3-small model

### Step 2: Search for Similar Emails

Submit any email file to find the top 5 most similar emails by content:

```bash
python email_similarity.py --search smaller_email/sample-1.eml --top 5
```

**Output:**
- Shows top 5 similar emails
- Displays only the EMAIL BODY content
- Shows similarity scores (0-1, higher = more similar)
- Automatically deduplicates results

## How It Works

1. **Email Parsing**: Extracts subject, sender, URLs, body text from EML files
2. **Chunking**: Splits email content into 1000-char chunks with 200-char overlap using RecursiveCharacterTextSplitter
3. **Embeddings**: Creates vector embeddings using OpenAI's text-embedding-3-small model
4. **Storage**: Stores in local ChromaDB vector database
5. **Similarity Search**: Uses cosine similarity to find most similar email content

## Customization

### Search different number of results
```bash
python email_similarity.py --search smaller_email/sample-1.eml --top 10
```

### Use different email directory
```bash
python email_similarity.py --build --email-dir email
```

### Use different database location
```bash
python email_similarity.py --build --db-dir ./my_custom_db
python email_similarity.py --search email.eml --db-dir ./my_custom_db
```

## Technical Details

- **Embedding Model**: text-embedding-3-small (OpenAI)
- **Vector Store**: ChromaDB (local, persistent)
- **Chunking**: RecursiveCharacterTextSplitter (1000/200)
- **Similarity Metric**: Cosine distance (converted to similarity score)
- **Framework**: LangChain

## Example Output

```
================================================================================
TOP 5 SIMILAR EMAILS - EMAIL BODY ONLY
================================================================================

================================================================================
EMAIL #1
================================================================================
Similarity: 0.7953
Source: smaller_email/sample-1.eml
Subject: CLIENTE PRIME - BRADESCO LIVELO...
From: BANCO DO BRADESCO LIVELO <banco.bradesco@atendimento.com.br>

EMAIL BODY:
--------------------------------------------------------------------------------
Para visualizar as imagens deste email. Clique aqui Pontos Livelo...
================================================================================
```

## Notes

- The vector store only needs to be built once (unless emails change)
- Similarity scores closer to 1.0 indicate higher similarity
- The first result may be the query email itself if it exists in the database
- Emails are deduplicated by source file path

