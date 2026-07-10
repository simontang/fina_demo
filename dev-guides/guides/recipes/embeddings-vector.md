# Recipe: Custom Embeddings & VectorStore

Replace the embeddings provider or vector store backend.

## Files You'll Touch

| Step | File | Action |
|---|---|---|
| 1 | Your embeddings provider | Implement LangChain `Embeddings` |
| 2 | `EmbeddingsLatticeManager` | Register provider |
| 3 | Your vector store | Implement LangChain `VectorStore` |
| 4 | `VectorStoreLatticeManager` | Register store |

## Step 1: Custom Embeddings Provider

```typescript
// your-embeddings/CohereEmbeddingsProvider.ts
import { Embeddings, type EmbeddingsParams } from "@langchain/core/embeddings";

export class CohereEmbeddingsProvider extends Embeddings {
  private apiKey: string;

  constructor(params: EmbeddingsParams & { apiKey: string }) {
    super(params);
    this.apiKey = params.apiKey;
  }

  async embedDocuments(documents: string[]): Promise<number[][]> {
    const response = await fetch("https://api.cohere.ai/v1/embed", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        texts: documents,
        model: "embed-english-v3.0",
        input_type: "search_document",
      }),
    });

    const data = await response.json();
    return data.embeddings;
  }

  async embedQuery(text: string): Promise<number[]> {
    const response = await fetch("https://api.cohere.ai/v1/embed", {
      method: "POST",
      headers: {
        Authorization: `Bearer ${this.apiKey}`,
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        texts: [text],
        model: "embed-english-v3.0",
        input_type: "search_query",
      }),
    });

    const data = await response.json();
    return data.embeddings[0];
  }
}
```

## Step 2: Register Embeddings Provider

```typescript
import { registerEmbeddingsLattice } from "@axiom-lattice/core";
// Convenience function: registerEmbeddingsLattice(key, embeddings)
registerEmbeddingsLattice("cohere", new CohereEmbeddingsProvider({
  apiKey: process.env.COHERE_API_KEY!,
}));
// Or via manager instance: EmbeddingsLatticeManager.getInstance().registerLattice(key, embeddings)
```

## Step 3: Custom VectorStore

```typescript
// your-vectorstore/PineconeVectorStore.ts
import { VectorStore } from "@langchain/core/vectorstores";
import type { Embeddings } from "@langchain/core/embeddings";
import type { Document } from "@langchain/core/documents";

export class PineconeVectorStore extends VectorStore {
  _vectorstoreType(): string {
    return "pinecone";
  }

  constructor(embeddings: Embeddings, private pineconeIndex: any) {
    super(embeddings, {});
  }

  async addVectors(
    vectors: number[][],
    documents: Document[],
    options?: { ids?: string[] }
  ): Promise<string[]> {
    const ids = options?.ids || documents.map(() => crypto.randomUUID());
    const records = vectors.map((values, i) => ({
      id: ids[i],
      values,
      metadata: documents[i].metadata,
    }));

    await this.pineconeIndex.upsert(records);
    return ids;
  }

  async addDocuments(documents: Document[]): Promise<string[]> {
    const texts = documents.map((d) => d.pageContent);
    const vectors = await this.embeddings.embedDocuments(texts);
    return this.addVectors(vectors, documents);
  }

  async similaritySearchVectorWithScore(
    query: number[],
    k: number,
    filter?: any
  ): Promise<[Document, number][]> {
    const results = await this.pineconeIndex.query({
      vector: query,
      topK: k,
      filter,
      includeMetadata: true,
    });

    return results.matches.map((match: any) => [
      new Document({
        pageContent: match.metadata?.text || "",
        metadata: match.metadata,
      }),
      match.score,
    ]);
  }

  async delete(params: { ids?: string[] }): Promise<void> {
    if (params.ids) {
      await this.pineconeIndex.deleteMany(params.ids);
    }
  }

  static async fromDocuments(
    docs: Document[],
    embeddings: Embeddings,
    dbConfig: any
  ): Promise<PineconeVectorStore> {
    const store = new PineconeVectorStore(embeddings, dbConfig.index);
    await store.addDocuments(docs);
    return store;
  }

  static async fromExistingIndex(
    embeddings: Embeddings,
    dbConfig: any
  ): Promise<PineconeVectorStore> {
    return new PineconeVectorStore(embeddings, dbConfig.index);
  }
}
```

## Step 4: Register VectorStore

```typescript
import { VectorStoreLatticeManager } from "@axiom-lattice/core";

const mgr = VectorStoreLatticeManager.getInstance();
mgr.registerLattice("pinecone", pineconeVectorStore);
```

## Gotchas

- Embeddings and VectorStore must implement LangChain's interfaces (`@langchain/core/embeddings`, `@langchain/core/vectorstores`)
- `VectorStore` requires a static `fromDocuments` and `fromExistingIndex` factory method
- The embeddings provider is used by both the RAG pipeline and the vector store
- `similaritySearchVectorWithScore` returns `[Document, number]` — the number is the similarity score
