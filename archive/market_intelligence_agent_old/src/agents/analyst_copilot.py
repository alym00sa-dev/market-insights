"""
Analyst Copilot Agent

Natural language interface for competitive intelligence queries.

Design: Chat-based interface that routes queries to Competitive Reasoning
- Accepts natural language questions
- Parses intent and extracts parameters
- Routes to appropriate analysis method
- Formats responses for readability
- Maintains conversation memory

Why this design?
- Natural language queries are easier for analysts than programmatic API calls
- Intent parsing separates user interface from analysis logic
- Conversation memory enables multi-turn queries ("What about Anthropic?" after asking about OpenAI)
- Formatting makes insights actionable

Query types supported:
1. Event impact: "When X released Y, what happened?"
2. Provider comparison: "How do A and B differ on pillar X?"
3. Leadership ranking: "Who leads on pillar X?"
4. Timeline analysis: "How did pillar X evolve?"
"""

import json
import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional, Literal
from datetime import datetime, timedelta

from ..llm import LLMProvider
from ..storage import EventDatabase, EventVectorStore
from ..models import Pillar
from .competitive_reasoning import CompetitiveReasoning


class AnalystCopilot:
    """
    Natural language interface for competitive intelligence.

    Core capability: Transform user questions into structured analysis.
    """

    def __init__(
        self,
        llm_provider: LLMProvider,
        database: EventDatabase,
        vector_store: EventVectorStore,
        reasoning: CompetitiveReasoning,
        config: Dict[str, Any]
    ):
        """
        Initialize Analyst Copilot.

        Args:
            llm_provider: LLM for intent parsing (simple tasks)
            database: Event database
            vector_store: Vector store for semantic search
            reasoning: Competitive Reasoning agent
            config: Configuration dict
        """
        self.llm = llm_provider
        self.db = database
        self.vector_store = vector_store
        self.reasoning = reasoning
        self.config = config

        # Conversation memory
        self.conversation_history: List[Dict[str, str]] = []
        self.last_query_context: Dict[str, Any] = {}

        # Load Market Intelligence Glossary
        self.glossary = self._load_glossary()

    # ========================================================================
    # MAIN INTERFACE
    # ========================================================================

    def chat(
        self,
        query: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Chat interface that accepts start_date and end_date parameters.

        This is a convenience wrapper around query() that converts date ranges
        to time_filter_days for the UI.

        Args:
            query: Natural language question
            start_date: ISO format start date (YYYY-MM-DD) or None for all time
            end_date: ISO format end date (YYYY-MM-DD) or None for all time
            **kwargs: Additional parameters passed to query()

        Returns:
            Dict with 'response' and 'sources' keys for UI compatibility
        """
        # Convert start_date to time_filter_days
        time_filter_days = None
        if start_date and end_date:
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_date)
                end_dt = datetime.fromisoformat(end_date)
                time_filter_days = (end_dt - start_dt).days
            except (ValueError, TypeError):
                # If date parsing fails, use None (all time)
                time_filter_days = None
        elif start_date:
            # Only start_date provided, calculate days from start_date to now
            try:
                from datetime import datetime
                start_dt = datetime.fromisoformat(start_date)
                time_filter_days = (datetime.now() - start_dt).days
            except (ValueError, TypeError):
                time_filter_days = None

        # Call the existing query method
        result = self.query(
            user_query=query,
            time_filter_days=time_filter_days,
            **kwargs
        )

        # Convert response format for UI compatibility
        # UI expects 'response' key, but query() returns 'answer' key
        return {
            'response': result.get('answer', ''),
            'sources': result.get('sources', []),
            'usage': result.get('usage', {})
        }

    def query(
        self,
        user_query: str,
        include_raw_data: bool = False,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        search_mode: str = 'per_provider',
        num_sources: int = 10,
        response_length: str = 'medium',
        time_filter_days: Optional[int] = None,
        external_url: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a natural language query with conversational, ChatGPT-style responses.

        This is the main entry point for user interactions.

        Args:
            user_query: Natural language question
            include_raw_data: Whether to include raw analysis data in response
            max_tokens: Maximum tokens for response (controls response length)
            temperature: LLM temperature (0.0-1.0, controls creativity)
            search_mode: 'per_provider' (get results from each hyperscaler) or 'top_n' (get top N overall)
            num_sources: Number of source events to retrieve (default 10, can be 15 or 20)
            response_length: 'short', 'medium', or 'long' for response detail level
            time_filter_days: Number of days to filter by (None for all time)
            external_url: Optional URL to fetch and analyze external content

        Returns:
            Dict with:
                - answer: Natural, conversational response
                - sources: Event IDs cited
                - raw_data: Retrieved events (if requested)

        Examples:
            >>> copilot.query("When OpenAI released GPT-4 with 200K context, what happened?")
            >>> copilot.query("How do Anthropic and Google differ on alignment?")
            >>> copilot.query("Who leads on data pipelines?")
            >>> copilot.query("How does this compare? https://blog.google/...", external_url="https://...")
        """
        try:
            # Fetch external content if URL provided
            external_content = None
            if external_url:
                print(f"🔗 Attempting to fetch URL: {external_url}")
                external_content = self.fetch_url_content(external_url)
                if external_content and 'error' not in external_content:
                    print(f"✓ Successfully fetched external content:")
                    print(f"  Provider: {external_content.get('provider')}")
                    print(f"  Date: {external_content.get('date')}")
                    print(f"  Announcement: {external_content.get('announcement', '')[:100]}")
                else:
                    error_msg = external_content.get('error', 'Unknown error') if external_content else 'No content returned'
                    print(f"✗ Failed to fetch external content: {error_msg}")
                    external_content = None  # Don't use failed content
            # Retrieve relevant events from vector store
            if search_mode == 'per_provider':
                # Get results from each major hyperscaler for balanced coverage
                # Calculate results per provider: 10 sources = 2 per provider, 15 = 3, 20 = 4
                results_per_provider = max(2, num_sources // 5)
                relevant_events = self._search_per_provider(
                    user_query,
                    results_per_provider=results_per_provider,
                    max_total_results=num_sources,
                    time_filter_days=time_filter_days
                )
            else:
                # Get top N results overall
                relevant_events = self.vector_store.semantic_search(
                    query=user_query,
                    n_results=num_sources
                )

            # Build conversational prompt with context
            answer, sources, usage = self._generate_conversational_response(
                user_query,
                relevant_events,
                max_tokens=max_tokens,
                temperature=temperature,
                response_length=response_length,
                external_content=external_content,
                time_filter_days=time_filter_days
            )

            # Update conversation memory
            self._update_memory(user_query, answer)

            # Build response
            response = {
                'answer': answer,
                'sources': sources,
                'usage': usage  # Include token usage stats
            }

            if include_raw_data:
                response['raw_data'] = relevant_events

            return response

        except Exception as e:
            return {
                'answer': f"I encountered an error: {str(e)}. Could you try rephrasing your question?",
                'sources': []
            }

    def clear_memory(self):
        """Clear conversation history."""
        self.conversation_history = []
        self.last_query_context = {}

    def extract_url(self, query: str) -> Optional[str]:
        """
        Extract URL from user query if present.

        Args:
            query: User's question

        Returns:
            URL string if found, None otherwise
        """
        import re
        # Match http/https URLs
        url_pattern = r'https?://[^\s<>"{}|\\^`\[\]]+'
        matches = re.findall(url_pattern, query)
        return matches[0] if matches else None

    def fetch_url_content(self, url: str) -> Optional[Dict[str, str]]:
        """
        Fetch and extract content from a URL using actual web fetching.

        Args:
            url: URL to fetch

        Returns:
            Dict with extracted information (provider, announcement, date, summary)
        """
        try:
            # Import subprocess to call WebFetch
            import subprocess
            import json

            # Create a Python script that uses WebFetch-like functionality
            # Since we can't directly call WebFetch from here, we'll use the LLM with a proper prompt
            # that simulates fetching by having it extract from the actual URL content

            # For now, use a simple approach: fetch with requests if available
            try:
                import requests
                from bs4 import BeautifulSoup

                headers = {
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                }
                response = requests.get(url, headers=headers, timeout=10)
                response.raise_for_status()

                # Parse HTML
                soup = BeautifulSoup(response.text, 'html.parser')

                # Extract text content
                page_text = soup.get_text(separator=' ', strip=True)

                # Use LLM to extract structured info from the fetched content
                extraction_prompt = f"""Extract key information from this web page content.

URL: {url}

Page content:
{page_text[:8000]}

Please identify:
1. Provider/Company (e.g., OpenAI, Google, Anthropic, Meta, Microsoft, Snowflake, etc.)
2. What was announced or changed
3. Publication date (look for date in the content - be precise!)
4. Brief 2-3 sentence summary

Return ONLY valid JSON:
{{
    "provider": "CompanyName",
    "announcement": "Brief description",
    "date": "YYYY-MM-DD",
    "summary": "2-3 sentences"
}}"""

                messages = [
                    {"role": "system", "content": "Extract information from web content and return ONLY valid JSON."},
                    {"role": "user", "content": extraction_prompt}
                ]

                llm_response = self.llm.generate(
                    messages=messages,
                    task_complexity="simple",
                    temperature=0.1
                )

                content = llm_response['content'].strip()

                # Extract JSON if wrapped
                if '```' in content:
                    content = content.split('```')[1]
                    if content.startswith('json'):
                        content = content[4:]
                    content = content.strip()

                extracted = json.loads(content)
                extracted['url'] = url
                return extracted

            except ImportError:
                print("Warning: requests/beautifulsoup not available, using basic extraction")
                return {'error': 'Web fetching libraries not available', 'url': url}

        except Exception as e:
            print(f"Warning: Failed to fetch URL content: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e), 'url': url}

    def _load_glossary(self) -> Dict[str, Any]:
        """Load Market Intelligence Glossary for contextualization."""
        try:
            # Find glossary file (relative to this file's location)
            glossary_path = Path(__file__).parent.parent.parent / "glossary.yaml"

            if glossary_path.exists():
                with open(glossary_path) as f:
                    return yaml.safe_load(f)
            else:
                print(f"Warning: Glossary not found at {glossary_path}")
                return {}
        except Exception as e:
            print(f"Warning: Failed to load glossary: {e}")
            return {}

    def _get_length_instruction(self, response_length: str) -> str:
        """Get response length guidance based on user's selection."""
        if response_length == 'short':
            return """- Provide a CONCISE, focused response (aim for 100-200 words)
- Cover only the key points
- Be direct and efficient
- Prioritize the most critical information"""
        elif response_length == 'long':
            return """- Provide a COMPREHENSIVE, thorough response (aim for 1,000-1,200 words)
- Explore the topic in depth with detailed analysis
- Include multiple examples, context, and strategic implications
- Cover competitive dynamics and market trends
- Elaborate on major points with supporting evidence
- Think like writing a detailed analyst briefing"""
        else:  # medium/balanced
            return """- Provide a BALANCED, moderately detailed response (aim for 400-700 words)
- Cover the main points with good context and examples
- Include relevant analysis and implications
- Balance breadth and depth appropriately"""

    def _get_glossary_context(self, query: str) -> str:
        """Get relevant glossary entries based on query keywords."""
        if not self.glossary:
            return ""

        query_lower = query.lower()
        relevant_entries = []

        # Check if query is about benchmarks/performance/comparison - always include AA context
        benchmark_keywords = ['benchmark', 'performance', 'compare', 'best', 'score', 'intelligence index',
                             'coding index', 'math index', 'mmlu', 'gpqa', 'latest model', 'aa ']
        include_aa_context = any(keyword in query_lower for keyword in benchmark_keywords)

        # Add AA context if relevant
        if include_aa_context:
            data_sources = self.glossary.get('data_sources', {})
            aa_data = data_sources.get('artificial_analysis', {})
            if aa_data:
                aa_desc = aa_data.get('description', '')
                aa_why = aa_data.get('why_it_matters', '')
                relevant_entries.append(f"**Artificial Analysis (AA)**: {aa_desc} {aa_why}")

        # Check for benchmark mentions
        benchmarks = self.glossary.get('benchmarks', {})
        for bench_name, bench_data in benchmarks.items():
            if bench_name.replace('_', ' ') in query_lower or bench_data.get('full_name', '').lower() in query_lower:
                full_name = bench_data.get('full_name', bench_name)
                description = bench_data.get('description', '')
                why_matters = bench_data.get('why_it_matters', '')
                relevant_entries.append(f"**{full_name}**: {description} {why_matters}")

        # Check for market term mentions
        market_terms = self.glossary.get('market_terms', {})
        for term_name, term_data in market_terms.items():
            if term_name.replace('_', '-') in query_lower or term_name.replace('_', ' ') in query_lower:
                definition = term_data.get('definition', '')
                why_matters = term_data.get('why_it_matters', '')
                relevant_entries.append(f"**{term_name.replace('_', ' ').title()}**: {definition} {why_matters}")

        # Check for technical concept mentions
        technical = self.glossary.get('technical_concepts', {})
        for concept_name, concept_data in technical.items():
            if concept_name.replace('_', ' ') in query_lower:
                definition = concept_data.get('definition', '')
                why_matters = concept_data.get('why_it_matters', '')
                relevant_entries.append(f"**{concept_name.replace('_', ' ').title()}**: {definition} {why_matters}")

        if relevant_entries:
            return "\n\nRelevant definitions from Market Intelligence Glossary:\n" + "\n".join(relevant_entries[:4])  # Max 4 entries (AA + 3 others)
        return ""

    def _search_per_provider(self, user_query: str, results_per_provider: int = 2, max_total_results: int = 10, time_filter_days: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Search for relevant events, ensuring representation from each major hyperscaler.

        Time filtering behavior:
        - Specific time filter (30/90 days): Hard cutoff - only returns events within timeframe
        - All Time mode: 12-month recency bias (prefers recent) but falls back to older data if needed

        Args:
            user_query: User's question
            results_per_provider: Number of results to get per provider
            max_total_results: Maximum total results to return
            time_filter_days: Number of days for time filter, None for All Time mode

        Returns:
            List of relevant events with balanced provider representation
        """
        from datetime import datetime, timedelta

        # Major AI hyperscalers to ensure coverage
        hyperscalers = ['OpenAI', 'Anthropic', 'Google', 'Microsoft', 'Meta']

        all_events = []
        seen_event_ids = set()

        # Calculate cutoff dates based on time filter mode
        # - Specific time filter (30/90 days): Hard cutoff, strict filtering
        # - All Time mode: Recency bias (prefer last 12 months) but fall back to older data
        is_all_time = time_filter_days is None

        if is_all_time:
            # All Time mode: Apply 12-month recency bias for first pass
            recency_bias_days = 365  # 12 months
            recency_cutoff = datetime.now() - timedelta(days=recency_bias_days)
            recency_cutoff_iso = recency_cutoff.isoformat()
            hard_cutoff_iso = None  # No hard cutoff in All Time mode
            print(f"🔍 Time Filter - All Time (12-month recency bias, will fall back to older data if needed)")
        else:
            # Specific time filter: Hard cutoff, no fallback
            hard_cutoff = datetime.now() - timedelta(days=time_filter_days)
            hard_cutoff_iso = hard_cutoff.isoformat()
            recency_cutoff_iso = hard_cutoff_iso  # Same as hard cutoff
            print(f"🔍 Time Filter - {time_filter_days} days (hard cutoff: {hard_cutoff.strftime('%Y-%m-%d')})")

        # First pass: Try to get recent events from each provider
        for provider in hyperscalers:
            try:
                # Search with provider filter - get more results to filter by date
                provider_events = self.vector_store.semantic_search(
                    query=user_query,
                    n_results=results_per_provider * 3,  # Get more to filter
                    provider=provider
                )

                # Filter events with recency bias
                recent_events = []
                older_events = []  # For fallback in All Time mode

                for event in provider_events:
                    event_id = event.get('event_id')
                    published_at = event.get('metadata', {}).get('published_at', '')

                    if event_id and event_id not in seen_event_ids:
                        # Hard time filter: strict filtering, skip events outside window
                        if hard_cutoff_iso and published_at < hard_cutoff_iso:
                            continue

                        # Recency bias: prefer recent, but keep older for fallback
                        if published_at >= recency_cutoff_iso:
                            recent_events.append(event)
                            seen_event_ids.add(event_id)
                        elif is_all_time:
                            # In All Time mode, keep older events as fallback
                            older_events.append(event)

                        if len(recent_events) >= results_per_provider:
                            break

                # Use recent events if available, otherwise fall back to older (All Time only)
                if recent_events:
                    all_events.extend(recent_events[:results_per_provider])
                    print(f"  ✓ Added {len(recent_events[:results_per_provider])} recent events")
                elif is_all_time and older_events:
                    # Fall back to older events in All Time mode
                    for event in older_events[:results_per_provider]:
                        all_events.append(event)
                        seen_event_ids.add(event.get('event_id'))
                    print(f"  ✓ Added {len(older_events[:results_per_provider])} older events (fallback)")
                else:
                    print(f"  ✗ No events added (recent: {len(recent_events)}, older: {len(older_events)})")

            except Exception as e:
                # If provider filtering fails or no results, continue
                continue

        # If we still don't have enough results, do a general search
        if len(all_events) < 10:
            general_results = self.vector_store.semantic_search(
                query=user_query,
                n_results=20  # Get more to filter by date
            )

            # Apply same recency bias logic
            for event in general_results:
                event_id = event.get('event_id')
                published_at = event.get('metadata', {}).get('published_at', '')

                if event_id and event_id not in seen_event_ids:
                    # Hard time filter: strict filtering
                    if hard_cutoff_iso and published_at < hard_cutoff_iso:
                        continue

                    # Add event (recency bias naturally prioritizes recent in All Time mode)
                    all_events.append(event)
                    seen_event_ids.add(event_id)
                    if len(all_events) >= max_total_results:
                        break

        # Sort by date (most recent first) before returning
        all_events.sort(
            key=lambda x: x.get('metadata', {}).get('published_at', ''),
            reverse=True
        )

        print(f"\n📋 Final events before returning ({len(all_events[:max_total_results])} events):")
        for i, event in enumerate(all_events[:max_total_results], 1):
            provider = event.get('metadata', {}).get('provider', 'Unknown')
            date = event.get('metadata', {}).get('published_at', 'Unknown')[:10]
            print(f"  [{i}] {provider} ({date})")

        return all_events[:max_total_results]

    # ========================================================================
    # CONVERSATIONAL RESPONSE GENERATION
    # ========================================================================

    def _generate_conversational_response(
        self,
        user_query: str,
        relevant_events: List[Dict[str, Any]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        response_length: str = 'medium',
        external_content: Optional[Dict[str, str]] = None,
        time_filter_days: Optional[int] = None
    ) -> tuple[str, List[str], Dict[str, int]]:
        """
        Generate a natural, conversational response like ChatGPT with inline citations.

        Args:
            user_query: User's question
            relevant_events: Relevant events from vector store
            max_tokens: Maximum tokens for response (controls response length)
            temperature: LLM temperature (0.0-1.0, controls creativity)
            response_length: 'short' (concise), 'medium' (balanced), 'long' (comprehensive)
            external_content: Optional external URL content for comparison

        Returns:
            Tuple of (conversational_answer_with_citations, source_event_ids, token_usage)
        """
        # Build context from relevant events with numbered sources
        context_parts = []
        source_ids = []
        source_details = []

        for i, event_data in enumerate(relevant_events[:10], 1):
            event_id = event_data.get('event_id')
            provider = event_data.get('metadata', {}).get('provider', 'Unknown')
            if event_id:
                event = self.db.get_event(event_id)
                if event:
                    print(f"  ✓ Retrieved event {i}: {provider}")
                    source_ids.append(event_id)
                    # Store source details for footnote
                    source_details.append({
                        'num': i,
                        'provider': event.provider,
                        'what_changed': event.what_changed,
                        'source_url': event.source_url,
                        'published_at': event.published_at.strftime('%Y-%m-%d')
                    })
                    # Add numbered source to context
                    context_parts.append(
                        f"[{i}] {event.provider}: {event.what_changed} "
                        f"(Published: {event.published_at.strftime('%Y-%m-%d')})"
                    )
                else:
                    print(f"  ✗ Failed to retrieve event {i}: {provider} (event_id: {event_id})")
            else:
                print(f"  ✗ No event_id for event {i}: {provider}")

        context_text = "\n".join(context_parts) if context_parts else "No specific events found."

        # Build conversation history for context
        history_text = ""
        if self.conversation_history:
            recent_history = self.conversation_history[-6:]  # Last 3 turns
            for msg in recent_history:
                role = msg['role'].title()
                content = msg['content'][:200]  # Truncate
                history_text += f"{role}: {content}\n"

        # Get relevant glossary context for this query
        glossary_context = self._get_glossary_context(user_query)

        # Build external content context if provided
        external_context = ""
        if external_content and 'error' not in external_content:
            provider = external_content.get('provider', 'Unknown')
            announcement = external_content.get('announcement', 'N/A')
            date = external_content.get('date', 'Unknown date')
            summary = external_content.get('summary', 'N/A')
            url = external_content.get('url', '')

            # Add note if date is unclear
            date_note = ""
            if date in ['Unknown date', 'null', None, '']:
                date_note = " (Note: Publication date not clearly stated on page - verify independently if needed)"

            external_context = f"""

EXTERNAL ANNOUNCEMENT TO ANALYZE:
Provider: {provider}
Date: {date}{date_note}
Announcement: {announcement}
Summary: {summary}
URL: {url}

COMPARISON INSTRUCTIONS:
The user has provided an external announcement to compare with events in our database. Please:
1. Summarize what this external announcement is about
2. Search our database for similar or related developments in the specified timeframe
3. Compare and contrast:
   - Is this announcement novel or part of an existing trend?
   - How does it compare to what other hyperscalers have done?
   - What makes this significant or different?
4. Provide competitive context: Who else is doing similar things? Who's ahead/behind?
5. If relevant, note any strategic implications or market shifts

Structure your response as:
1. Brief summary of the external announcement
2. Comparison with database events (cite with [1], [2], etc.)
3. Competitive analysis and implications
"""

        # Create conversational system prompt with citation instructions
        system_prompt = """You are a professional market intelligence analyst specializing in frontier AI and competitive dynamics.

Your role is to provide clear, accessible analysis of AI industry developments. Write in a professional but conversational tone—like a business analyst presenting findings to stakeholders. Be authoritative and well-informed while remaining readable and engaging.

Data Sources:
When you reference benchmark data (Intelligence Index, Coding Index, MMLU, GPQA, etc.), this comes from Artificial Analysis (AA), an independent platform that conducts standardized testing of AI models. AA provides vendor-neutral, third-party validation that enables fair comparison across OpenAI, Anthropic, Google, Meta, and other providers. When appropriate, briefly explain what AA metrics mean in context (e.g., "AA Intelligence Index measures overall reasoning capability").

Key guidelines:
- Write in a professional, business-appropriate tone
- Be clear and accessible without being overly casual
- Use precise language and specific details
- Avoid fluff—get straight to the point with substantive information
- Lead with facts, data, and concrete examples
- Skip hedging language ("it seems", "perhaps", "it appears") unless genuinely uncertain
- CRITICAL: ONLY discuss information explicitly present in the numbered sources [1], [2], [3], etc.
- DO NOT use your pre-trained knowledge to discuss companies or events not in the provided sources
- If asked about a hyperscaler but no sources exist for them, state "No recent data available for [Company]"
- Amazon, Nvidia, and other companies are NOT tracked hyperscalers - only mention if they appear in source comparisons
- CRITICAL FORMATTING RULES:
- Write in plain text with normal spaces between words - DO NOT remove spaces or compress text
- Use regular hyphens (-) not math symbols (−)
- Use dollar signs normally: "$80 billion" not special formatting
- DO NOT use LaTeX, math mode, or special encoding
- Use ONLY simple numbered brackets like [1], [2], [3] for citations
- Cite sources naturally within sentences, e.g., "OpenAI released GPT-4 with 128k context [1], which prompted Google to..."
- Multiple sources can be cited together like [1][2] or [1,2]
- Reference specific providers, dates, and developments with precision
- Identify and articulate patterns, strategic implications, and competitive dynamics
- Maintain a professional analytical tone throughout
- If information is limited, acknowledge gaps objectively
- DO NOT add a sources section at the end - the system will handle that
- When discussing benchmarks or market terms, use the glossary definitions provided for accuracy

CRITICAL - Avoid these citation mistakes:
- NEVER create markdown hyperlinks like [text](url) in your response body
- NEVER embed URLs directly in the text
- NEVER use complex citation formats - ONLY use [1], [2], [3]
- DO NOT create footnote-style citations - just use the numbers

Also avoid:
- Formal report structures with headers and bullet points
- Overly structured markdown formatting
- Fluff language and filler phrases ("it's worth noting", "interestingly", "as we can see")
- Hedging unnecessarily ("seems like", "appears to be", "might suggest")
- Repetitive phrasing or stating the obvious
- Vague generalizations without specific data
- Adding your own "Sources:" or "References:" section (the system handles this)"""

        # Determine time context for prompt
        if time_filter_days is None:
            time_context = "Based on all available market intelligence data (all time, with emphasis on recent developments)"
        elif time_filter_days <= 30:
            time_context = f"Based on market intelligence from the last {time_filter_days} days"
        elif time_filter_days <= 90:
            time_context = f"Based on market intelligence from the last ~{time_filter_days} days"
        else:
            time_context = f"Based on market intelligence from the specified timeframe"

        # Build user prompt with context
        user_prompt = f"""{time_context}, please answer this question naturally:

{user_query}
{external_context}

Relevant numbered sources from our database:
{context_text}
{glossary_context}

CRITICAL SOURCE ADHERENCE - READ CAREFULLY:
- ONLY use information from the numbered sources [1], [2], [3] above
- DO NOT use pre-trained knowledge about companies, models, or events
- If a hyperscaler has no sources listed above, do NOT discuss them or state "No recent data available"
- Amazon is NOT a tracked hyperscaler - only mention if it appears in a source comparison
- Every factual claim must be supported by a citation [1], [2], [3]

CRITICAL FORMATTING REQUIREMENTS - READ CAREFULLY:
1. Write in plain text with NORMAL SPACES between EVERY word
2. Use regular hyphens (-), not math symbols (−)
3. Dollar amounts: "$80 billion" with space, NOT "$80billion" compressed
4. Numbers with words: "50 million" with space, NOT "50million"
5. DO NOT use LaTeX, math mode, or any text encoding that removes spaces
6. DO NOT let words run together - every word must have a space after it
7. Citations: Use ONLY simple numbered brackets [1], [2], [3]
8. Example with SPACES: "Microsoft announced $90 billion in investments [1] and Google responded with..."
9. BAD example WITHOUT spaces: "$90billioninvestments" - NEVER do this
10. DO NOT create hyperlinks like [text](url) in your response body
11. DO NOT embed URLs in your response body

RESPONSE LENGTH GUIDANCE:
{self._get_length_instruction(response_length)}

CRITICAL - When Time Filter Returns NO Results:
If the search returns no events within the user's specified timeframe (e.g., "last 60 days"):
1. State clearly: "We don't have [topic] announcements in the [last 60 days]"
2. List what DOES exist using BULLET POINTS (3-5 most recent items from outside the timeframe):
   • Provider - Brief description (Date)
   • Provider - Brief description (Date)
3. End with: "Were you looking for something more specific?"
4. IGNORE ALL LENGTH REQUIREMENTS - keep it short (100-200 words max)
5. DO NOT elaborate or pad the response to meet word count targets

Example format:
"We don't have persistent memory announcements in the last 60 days. The most recent activity was:
• Anthropic - Automatic Memory for Teams (March 15, 2025)
• Google - Gemini Personal Context launch (February 15, 2025)
• Meta - Llama 4 preview with memory APIs (February 15, 2025)

Were you looking for something more specific?"

Please provide a conversational, helpful response with proper spacing and formatting that addresses the question using the available information."""

        # Add conversation history if available
        if history_text:
            user_prompt = f"""Recent conversation context:
{history_text}

{user_prompt}"""

        # Generate response using LLM
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        # Use provided temperature or default to 0.7 for natural responses
        response_temperature = temperature if temperature is not None else 0.7

        response = self.llm.generate(
            messages=messages,
            task_complexity="complex",  # Use Claude for natural dialogue
            temperature=response_temperature,  # User-configurable creativity
            max_tokens=max_tokens  # User-configurable response length
        )

        answer = response['content'].strip()

        # Capture token usage for tracking
        usage = response.get('usage', {})
        input_tokens = usage.get('input_tokens', 0)
        output_tokens = usage.get('output_tokens', 0)

        # Log token usage
        print(f"🔢 Token Usage - Input: {input_tokens}, Output: {output_tokens}, Total: {input_tokens + output_tokens}")

        # Add footnote-style source list at the end with hyperlinked text
        if source_details:
            answer += "\n\n**Sources:**\n"
            for source in source_details:
                import re

                # Check if this is a benchmark event from Artificial Analysis
                is_benchmark = 'artificialanalysis.ai' in source['source_url'].lower()

                if is_benchmark:
                    # For benchmarks, create clean title from what_changed
                    # Extract model name from the beginning
                    what_changed = source['what_changed']
                    if 'model' in what_changed.lower():
                        # Extract: "OpenAI model 'GPT-5.2 (xhigh)'" -> "OpenAI GPT-5.2 (xhigh)"
                        model_match = re.search(r"(.*?)\s+model\s+'([^']+)'", what_changed)
                        if model_match:
                            provider = model_match.group(1).strip()
                            model_name = model_match.group(2).strip()
                            title = f"{provider} {model_name} - Benchmark data"
                        else:
                            title = f"{source['provider']} model benchmark data"
                    else:
                        title = f"{source['provider']} model benchmark data"
                else:
                    # For regular events, clean up text
                    title = source['what_changed'][:100]
                    # Remove markdown formatting artifacts
                    title = re.sub(r'\*\*', '', title)  # Remove bold markers
                    title = re.sub(r'##\s*', '', title)  # Remove headers
                    # Fix common text compression issues
                    title = re.sub(r'([a-z])([A-Z])', r'\1 \2', title)
                    title = re.sub(r'([a-zA-Z])(\d)', r'\1 \2', title)
                    title = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', title)

                    if len(source['what_changed']) > 100:
                        title += "..."

                # Format: [1] hyperlinked text
                link_text = f"{title} - {source['provider']} ({source['published_at']})"
                answer += f"[{source['num']}] [{link_text}]({source['source_url']})\n"

        return answer, source_ids, usage

    # ========================================================================
    # INTENT PARSING (Legacy - kept for potential future use)
    # ========================================================================

    def _parse_intent(self, user_query: str) -> Dict[str, Any]:
        """
        Parse user query to determine intent and extract parameters.

        Uses LLM (simple task) to classify query type and extract entities.

        Returns:
            Dict with:
                - query_type: "event_impact", "provider_comparison", "leadership_ranking", "timeline"
                - parameters: Extracted parameters (providers, pillar, dates, event_id)
        """
        # Get available providers and pillars
        providers = self.db.get_all_providers()
        pillars = [p.value for p in Pillar]

        prompt = f"""Parse this competitive intelligence query and extract structured information.

User Query: "{user_query}"

Available Providers: {', '.join(providers)}
Available Pillars: {', '.join(pillars)}

Determine:
1. Query Type:
   - "event_impact" = asking about specific event's impact
   - "provider_comparison" = comparing providers
   - "leadership_ranking" = asking who leads
   - "timeline" = asking how something evolved over time

2. Parameters:
   - providers: List of provider names mentioned (empty if not specified)
   - pillar: Pillar name if mentioned (null if not specified)
   - event_id: Event ID if mentioned (null if not specified)
   - start_date: Start date if mentioned as "YYYY-MM-DD" (null if not specified)
   - end_date: End date if mentioned as "YYYY-MM-DD" (null if not specified)
   - time_range_days: If relative time mentioned (e.g., "last 6 months" = 180)

Context from previous query (use if user says "what about...", "how about..."):
{json.dumps(self.last_query_context) if self.last_query_context else "None"}

Return ONLY a JSON object with:
{{
  "query_type": "event_impact|provider_comparison|leadership_ranking|timeline",
  "parameters": {{
    "providers": ["Provider1", "Provider2"],
    "pillar": "PILLAR_NAME or null",
    "event_id": "evt_... or null",
    "start_date": "YYYY-MM-DD or null",
    "end_date": "YYYY-MM-DD or null",
    "time_range_days": 180
  }}
}}"""

        messages = [
            {"role": "system", "content": "You are a query parser for competitive intelligence. Return ONLY valid JSON, no other text."},
            {"role": "user", "content": prompt}
        ]

        try:
            response = self.llm.generate(
                messages=messages,
                task_complexity="simple",  # GPT-4 for simple parsing
                temperature=0.1
            )

            content = response['content'].strip()

            # Extract JSON if wrapped
            if content.startswith('```'):
                content = content.split('```')[1]
                if content.startswith('json'):
                    content = content[4:]
                content = content.strip()

            intent = json.loads(content)
            return intent

        except Exception as e:
            return {'error': f'Failed to parse query intent: {str(e)}'}

    # ========================================================================
    # ANALYSIS EXECUTION
    # ========================================================================

    def _execute_analysis(self, intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute appropriate analysis based on parsed intent.

        Routes to correct Competitive Reasoning method.
        """
        query_type = intent['query_type']
        params = intent['parameters']

        try:
            # Convert string dates to datetime
            start_date = datetime.fromisoformat(params['start_date']) if params.get('start_date') else None
            end_date = datetime.fromisoformat(params['end_date']) if params.get('end_date') else None

            # Apply relative time range if specified
            if params.get('time_range_days') and not start_date:
                start_date = datetime.now() - timedelta(days=params['time_range_days'])

            # Convert pillar string to enum
            pillar = None
            if params.get('pillar'):
                try:
                    pillar = Pillar(params['pillar'])
                except ValueError:
                    # Try case-insensitive match
                    for p in Pillar:
                        if p.value.lower() == params['pillar'].lower():
                            pillar = p
                            break

            # Route to appropriate method
            if query_type == "event_impact":
                event_id = params.get('event_id')
                if not event_id:
                    # Try to find event from description
                    return {'error': 'Event ID required for impact analysis. Please specify the event.'}

                return self.reasoning.analyze_event_impact(event_id)

            elif query_type == "provider_comparison":
                providers = params.get('providers', [])
                if len(providers) < 2:
                    # Get all providers if not specified
                    providers = self.db.get_all_providers()[:3]  # Compare top 3

                return self.reasoning.compare_providers(
                    providers=providers,
                    pillar=pillar,
                    start_date=start_date,
                    end_date=end_date
                )

            elif query_type == "leadership_ranking":
                if not pillar:
                    return {'error': 'Pillar required for leadership ranking. Please specify which pillar (e.g., TECHNICAL_CAPABILITIES).'}

                return self.reasoning.rank_leadership(
                    pillar=pillar,
                    start_date=start_date,
                    end_date=end_date,
                    providers=params.get('providers')
                )

            elif query_type == "timeline":
                if not pillar:
                    return {'error': 'Pillar required for timeline analysis. Please specify which pillar.'}

                return self.reasoning.analyze_timeline(
                    pillar=pillar,
                    start_date=start_date,
                    end_date=end_date,
                    providers=params.get('providers')
                )

            else:
                return {'error': f'Unknown query type: {query_type}'}

        except Exception as e:
            return {'error': f'Analysis execution failed: {str(e)}'}

    # ========================================================================
    # RESPONSE FORMATTING
    # ========================================================================

    def _format_response(self, analysis: Dict[str, Any], intent: Dict[str, Any]) -> str:
        """
        Format analysis results into user-friendly text.

        Converts structured analysis into readable markdown.
        """
        query_type = analysis.get('query_type', intent['query_type'])

        if query_type == 'event_impact':
            return self._format_event_impact(analysis)
        elif query_type == 'provider_comparison':
            return self._format_provider_comparison(analysis)
        elif query_type == 'leadership_ranking':
            return self._format_leadership_ranking(analysis)
        elif query_type == 'timeline':
            return self._format_timeline(analysis)
        else:
            return json.dumps(analysis, indent=2)

    def _format_event_impact(self, analysis: Dict[str, Any]) -> str:
        """Format event impact analysis."""
        parts = []

        # Header
        event = analysis.get('event', {})
        parts.append(f"# Event Impact: {event.get('provider', 'Unknown')} - {analysis.get('event_id', '')}")
        parts.append("")

        # Immediate impact
        parts.append("## Immediate Impact")
        parts.append(analysis.get('immediate_impact', 'N/A'))
        parts.append("")

        # Competitive shifts
        shifts = analysis.get('competitive_shifts', [])
        if shifts:
            parts.append("## Competitive Shifts")
            for shift in shifts[:5]:
                parts.append(f"### {shift.get('dimension', 'N/A')}")
                parts.append(f"- {shift.get('shift', 'N/A')}")
                parts.append(f"- Evidence: {shift.get('evidence', 'N/A')}")
                parts.append("")

        # Triggered responses
        responses = analysis.get('triggered_responses', [])
        if responses:
            parts.append("## Triggered Responses")
            for resp in responses:
                parts.append(f"- **{resp.get('provider', 'Unknown')}**: {resp.get('response_event', 'N/A')} ({resp.get('timing', 'unknown timing')})")
            parts.append("")

        # Long-term implications
        parts.append("## Long-term Implications")
        parts.append(analysis.get('long_term_implications', 'N/A'))
        parts.append("")

        # Confidence
        parts.append(f"*Confidence: {analysis.get('confidence', 0):.0%}*")

        return "\n".join(parts)

    def _format_provider_comparison(self, analysis: Dict[str, Any]) -> str:
        """Format provider comparison."""
        parts = []

        # Header
        providers = analysis.get('providers', [])
        pillar = analysis.get('pillar', 'overall')
        parts.append(f"# Provider Comparison: {' vs '.join(providers)}")
        parts.append(f"*Focus: {pillar}*")
        parts.append("")

        # Key differences
        differences = analysis.get('differences', [])
        if differences:
            parts.append("## Key Differences")
            for diff in differences:
                parts.append(f"### {diff.get('dimension', 'N/A')}")
                positions = diff.get('provider_positions', {})
                for provider, position in positions.items():
                    parts.append(f"- **{provider}**: {position}")
                parts.append("")

        # Convergence/Divergence
        parts.append("## Convergence/Divergence")
        parts.append(analysis.get('convergence_divergence', 'N/A'))
        parts.append("")

        # Strategic implications
        parts.append("## Strategic Implications")
        parts.append(analysis.get('strategic_implications', 'N/A'))
        parts.append("")

        # Confidence
        parts.append(f"*Confidence: {analysis.get('confidence', 0):.0%}*")

        return "\n".join(parts)

    def _format_leadership_ranking(self, analysis: Dict[str, Any]) -> str:
        """Format leadership ranking."""
        parts = []

        # Header
        pillar = analysis.get('pillar', 'N/A')
        parts.append(f"# Leadership Ranking: {pillar}")
        parts.append("")

        # Rankings
        rankings = analysis.get('rankings', [])
        if rankings:
            parts.append("## Rankings")
            for i, ranking in enumerate(rankings, 1):
                provider = ranking.get('provider', 'Unknown')
                score = ranking.get('score', 0)
                parts.append(f"### {i}. {provider} - {score:.0f}/100")

                strengths = ranking.get('key_strengths', [])
                if strengths:
                    parts.append("**Strengths:**")
                    for s in strengths:
                        parts.append(f"- {s}")

                weaknesses = ranking.get('key_weaknesses', [])
                if weaknesses:
                    parts.append("**Weaknesses:**")
                    for w in weaknesses:
                        parts.append(f"- {w}")

                evidence = ranking.get('evidence_events', [])
                if evidence:
                    parts.append(f"*Evidence: {len(evidence)} events*")

                parts.append("")

        # Analysis
        parts.append("## Analysis")
        parts.append(analysis.get('analysis', 'N/A'))
        parts.append("")

        # Confidence
        parts.append(f"*Confidence: {analysis.get('confidence', 0):.0%}*")

        return "\n".join(parts)

    def _format_timeline(self, analysis: Dict[str, Any]) -> str:
        """Format timeline analysis."""
        parts = []

        # Header
        pillar = analysis.get('pillar', 'N/A')
        parts.append(f"# Timeline Analysis: {pillar}")
        parts.append("")

        # Narrative
        parts.append("## Narrative")
        parts.append(analysis.get('narrative', 'N/A'))
        parts.append("")

        # Key trends
        trends = analysis.get('key_trends', [])
        if trends:
            parts.append("## Key Trends")
            for trend in trends:
                parts.append(f"- {trend}")
            parts.append("")

        # Timeline (sample)
        timeline = analysis.get('timeline', [])
        if timeline:
            parts.append("## Timeline")
            for entry in timeline[:10]:
                date = entry.get('date', 'N/A')
                provider = entry.get('provider', 'Unknown')
                description = entry.get('description', 'N/A')
                parts.append(f"- **{date}** | {provider}: {description}")
            if len(timeline) > 10:
                parts.append(f"  *...and {len(timeline) - 10} more events*")
            parts.append("")

        # Turning points
        turning_points = analysis.get('turning_points', [])
        if turning_points:
            parts.append("## Turning Points")
            for tp in turning_points:
                event_id = tp.get('event_id', 'N/A')
                why = tp.get('why', 'N/A')
                parts.append(f"- **{event_id}**: {why}")
            parts.append("")

        # Future trajectory
        parts.append("## Future Trajectory")
        parts.append(analysis.get('future_trajectory', 'N/A'))
        parts.append("")

        # Confidence
        parts.append(f"*Confidence: {analysis.get('confidence', 0):.0%}*")

        return "\n".join(parts)

    # ========================================================================
    # MEMORY & CONTEXT
    # ========================================================================

    def _update_memory(self, user_query: str, response: str):
        """Update conversation memory."""
        self.conversation_history.append({
            'role': 'user',
            'content': user_query,
            'timestamp': datetime.now().isoformat()
        })

        self.conversation_history.append({
            'role': 'assistant',
            'content': response[:500],  # Store truncated response
            'timestamp': datetime.now().isoformat()
        })

        # Keep only last 10 turns
        if len(self.conversation_history) > 20:
            self.conversation_history = self.conversation_history[-20:]

    def _extract_sources(self, analysis: Dict[str, Any]) -> List[str]:
        """Extract event IDs cited in analysis."""
        sources = set()

        # Event impact
        if analysis.get('event_id'):
            sources.add(analysis['event_id'])

        # Competitive shifts
        for shift in analysis.get('competitive_shifts', []):
            if shift.get('evidence'):
                # Try to extract event IDs from evidence text
                pass

        # Rankings
        for ranking in analysis.get('rankings', []):
            for event_id in ranking.get('evidence_events', []):
                sources.add(event_id)

        # Timeline
        for entry in analysis.get('timeline', []):
            if entry.get('event_id'):
                sources.add(entry['event_id'])

        return sorted(sources)
