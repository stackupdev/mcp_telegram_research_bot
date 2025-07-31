def search_and_extract_papers(topic: str, max_results: int = 10) -> str:
    """
    Combined function that searches for papers and extracts detailed info for each.
    Returns beautifully formatted results with arXiv numbers and PDF URLs.
    """
    if not mcp_client_factory:
        return "❌ MCP client not initialized"
    
    try:
        # Step 1: Search for papers
        paper_ids = search_papers(topic, max_results)
        
        if not paper_ids:
            return f"📝 No papers found for topic '{topic}'. Try a different search term."
        
        # Step 2: Extract detailed info for each paper
        detailed_papers = []
        for paper_id in paper_ids:
            paper_info = extract_info(paper_id)
            if paper_info and not paper_info.startswith("There's no saved information"):
                try:
                    # Parse JSON if it's a string
                    if isinstance(paper_info, str):
                        paper_data = json.loads(paper_info)
                    else:
                        paper_data = paper_info
                    
                    detailed_papers.append({
                        'id': paper_id,
                        'data': paper_data
                    })
                except json.JSONDecodeError:
                    # If parsing fails, skip this paper
                    continue
        
        if not detailed_papers:
            return f"📄 Found {len(paper_ids)} papers for '{topic}' but couldn't extract detailed information. The papers may need to be processed first."
        
        # Step 3: Format results beautifully
        response = f"🔬 **Research Results for '{topic}'**\n"
        response += f"Found {len(detailed_papers)} papers with detailed information:\n\n"
        
        for i, paper in enumerate(detailed_papers, 1):
            paper_data = paper['data']
            paper_id = paper['id']
            
            response += f"**{i}. {paper_data.get('title', 'Unknown Title')}**\n"
            response += f"📋 **arXiv ID:** `{paper_id}` (tap to copy)\n"
            response += f"👥 **Authors:** {', '.join(paper_data.get('authors', [])[:3])}{'...' if len(paper_data.get('authors', [])) > 3 else ''}\n"
            response += f"📅 **Published:** {paper_data.get('published', 'Unknown')}\n"
            
            # Make PDF URL easily accessible
            pdf_url = paper_data.get('pdf_url', '')
            if pdf_url:
                response += f"📄 **PDF:** [Download Paper]({pdf_url})\n"
            
            # Add summary preview
            summary = paper_data.get('summary', '')
            if summary:
                # Truncate summary to first 200 characters
                summary_preview = summary[:200] + "..." if len(summary) > 200 else summary
                response += f"📖 **Summary:** {summary_preview}\n"
            
            response += "\n" + "─" * 40 + "\n\n"
        
        # Add helpful footer
        response += "💡 **Quick Actions:**\n"
        response += "• Tap any arXiv ID to copy it\n"
        response += "• Click PDF links to download papers\n"
        response += "• Use the buttons below for more research options"
        
        return response
        
    except Exception as e:
        print(f"Error in search_and_extract_papers: {e}")
        return f"❌ Error searching for papers on '{topic}': {str(e)}"
