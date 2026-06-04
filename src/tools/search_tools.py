"""
Search Tools available to agents.
 
Tools are the interface between agents and the external world.
Each tool has a clear input/output contract, meaning, agents call tools
without knowing their internal implementation.
 
Design decision: Tools return plain dicts, not domain objects.
This keeps tools decoupled from the agent layer and makes it
easy to serialize results into AgentState.
"""
 
from typing import List, Dict, Any, Optional
 
 
class VectorSearchTool:
    """
    Searches a FAISS vector store for relevant chunks.
 
    The primary retrieval tool for knowledge base queries.
    Returns chunks ranked by embedding similarity.
 
    Args:
        vector_store : FAISSVectorStore instance
        embedder     : Embedding model with embed_query() method
    """
 
    name = "vector_search"
    description = "Search the knowledge base for relevant documents"
 
    def __init__(self, vector_store, embedder):
        self.vector_store = vector_store
        self.embedder = embedder
 
    def search(
        self,
        query: str,
        k: int = 5,
        metadata_filter: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for relevant chunks.
 
        Args:
            query           : Search query
            k               : Number of results
            metadata_filter : Optional metadata filter
 
        Returns:
            List of dicts with content, source, score, metadata
        """
        try:
            query_embedding = self.embedder.embed_query(query)
            results = self.vector_store.search(
                query_embedding,
                k=k,
                metadata_filter=metadata_filter
            )
            return [
                {
                    "content": chunk.content,
                    "source": chunk.metadata.get("filename", chunk.chunk_id),
                    "score": float(score),
                    "metadata": chunk.metadata
                }
                for chunk, score in results
            ]
        except Exception as e:
            print(f"[VectorSearchTool] Search failed: {e}")
            return []
 
 
class WebSearchTool:
    """
    Searches the web for current information.
 
    Used when knowledge base doesn't contain the required
    information — particularly for real-time or recent data.
 
    In production: integrate with Tavily, SerpAPI, or Bing.
    Mock implementation used when no API key configured.
 
    Args:
        api_key    : Tavily or SerpAPI key (optional)
        llm_client : LLM for result summarization (optional)
    """
 
    name = "web_search"
    description = "Search the web for current information not in the knowledge base"
 
    def __init__(self, api_key: Optional[str] = None, llm_client=None):
        self.api_key = api_key
        self.llm_client = llm_client
 
    def search(
        self,
        query: str,
        k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Search the web for relevant results.
 
        Args:
            query : Search query
            k     : Number of results to return
 
        Returns:
            List of dicts with content, source, score, metadata
        """
        if self.api_key:
            return self._tavily_search(query, k)
        return self._mock_search(query, k)
 
    def _tavily_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Real web search via Tavily API."""
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=self.api_key)
            response = client.search(query, max_results=k)
            return [
                {
                    "content": r.get("content", ""),
                    "source": r.get("url", "web"),
                    "score": r.get("score", 0.5),
                    "metadata": {"url": r.get("url"), "title": r.get("title")}
                }
                for r in response.get("results", [])
            ]
        except Exception as e:
            print(f"[WebSearchTool] Tavily search failed: {e}")
            return self._mock_search(query, k)
 
    def _mock_search(self, query: str, k: int) -> List[Dict[str, Any]]:
        """Mock web search for testing without API key."""
        return [
            {
                "content": f"Mock web search result for: {query}. "
                           f"This would contain relevant web content in production.",
                "source": f"https://example.com/result-{i+1}",
                "score": 0.7 - (i * 0.05),
                "metadata": {"mock": True, "result_index": i}
            }
            for i in range(min(k, 3))
        ]
