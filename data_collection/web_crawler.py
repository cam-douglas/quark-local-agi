#!/usr/bin/env python3
"""
Web Crawler for Data Collection
Step 2: Data Collection & Preparation

This module provides web crawling capabilities for collecting training data
for the Quark AI Assistant model development framework.
"""

import asyncio
import aiohttp
import logging
from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import json
import hashlib
from urllib.parse import urljoin, urlparse
import re
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class CrawlResult:
    """Represents the result of a web crawl"""
    url: str
    content: str
    status: str  # "success", "error", "filtered"
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = None

class WebCrawler:
    """Web crawler for collecting training data"""
    
    def __init__(self, 
                 max_concurrent: int = 10,
                 request_delay: float = 1.0,
                 timeout: int = 30,
                 max_pages_per_domain: int = 100,
                 user_agent: str = "Quark-Data-Collector/1.0"):
        self.max_concurrent = max_concurrent
        self.request_delay = request_delay
        self.timeout = timeout
        self.max_pages_per_domain = max_pages_per_domain
        self.user_agent = user_agent
        
        self.session = None
        self.visited_urls: Set[str] = set()
        self.domain_pages: Dict[str, int] = {}
        self.results: List[CrawlResult] = []
        
        # Content filters
        self.content_filters = [
            r'<script[^>]*>.*?</script>',
            r'<style[^>]*>.*?</style>',
            r'<nav[^>]*>.*?</nav>',
            r'<footer[^>]*>.*?</footer>',
            r'<header[^>]*>.*?</header>'
        ]
        
        # Domain restrictions
        self.allowed_domains = [
            'wikipedia.org',
            'github.com',
            'stackoverflow.com',
            'docs.python.org',
            'pytorch.org',
            'tensorflow.org',
            'huggingface.co'
        ]
        
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={'User-Agent': self.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    def _is_allowed_domain(self, url: str) -> bool:
        """Check if domain is allowed for crawling"""
        try:
            domain = urlparse(url).netloc.lower()
            return any(allowed in domain for allowed in self.allowed_domains)
        except Exception:
            return False
    
    def _clean_content(self, html_content: str) -> str:
        """Clean HTML content and extract text"""
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', html_content)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)]', '', text)
        
        return text.strip()
    
    def _extract_metadata(self, url: str, html_content: str) -> Dict[str, Any]:
        """Extract metadata from HTML content"""
        metadata = {
            'url': url,
            'domain': urlparse(url).netloc,
            'content_length': len(html_content),
            'text_length': len(self._clean_content(html_content)),
            'timestamp': datetime.now().isoformat()
        }
        
        # Extract title
        title_match = re.search(r'<title[^>]*>(.*?)</title>', html_content, re.IGNORECASE)
        if title_match:
            metadata['title'] = title_match.group(1).strip()
        
        # Extract meta description
        desc_match = re.search(r'<meta[^>]*name=["\']description["\'][^>]*content=["\']([^"\']*)["\']', html_content, re.IGNORECASE)
        if desc_match:
            metadata['description'] = desc_match.group(1).strip()
        
        return metadata
    
    async def crawl_url(self, url: str) -> CrawlResult:
        """Crawl a single URL"""
        if url in self.visited_urls:
            return CrawlResult(url, "", "skipped", error="Already visited")
        
        # Check domain restrictions
        if not self._is_allowed_domain(url):
            return CrawlResult(url, "", "filtered", error="Domain not allowed")
        
        # Check domain page limits
        domain = urlparse(url).netloc
        if self.domain_pages.get(domain, 0) >= self.max_pages_per_domain:
            return CrawlResult(url, "", "filtered", error="Domain page limit reached")
        
        try:
            async with self.session.get(url) as response:
                if response.status == 200:
                    content = await response.text()
                    
                    # Clean content
                    cleaned_content = self._clean_content(content)
                    
                    # Skip if content is too short
                    if len(cleaned_content) < 100:
                        return CrawlResult(url, "", "filtered", error="Content too short")
                    
                    # Extract metadata
                    metadata = self._extract_metadata(url, content)
                    
                    # Update counters
                    self.visited_urls.add(url)
                    self.domain_pages[domain] = self.domain_pages.get(domain, 0) + 1
                    
                    # Add delay between requests
                    await asyncio.sleep(self.request_delay)
                    
                    return CrawlResult(
                        url=url,
                        content=cleaned_content,
                        status="success",
                        metadata=metadata,
                        timestamp=datetime.now().isoformat()
                    )
                else:
                    return CrawlResult(
                        url=url,
                        content="",
                        status="error",
                        error=f"HTTP {response.status}"
                    )
                    
        except Exception as e:
            return CrawlResult(
                url=url,
                content="",
                status="error",
                error=str(e)
            )
    
    async def crawl_urls(self, urls: List[str]) -> List[CrawlResult]:
        """Crawl multiple URLs concurrently"""
        semaphore = asyncio.Semaphore(self.max_concurrent)
        
        async def crawl_with_semaphore(url: str) -> CrawlResult:
            async with semaphore:
                return await self.crawl_url(url)
        
        tasks = [crawl_with_semaphore(url) for url in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle exceptions
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(CrawlResult(
                    url="unknown",
                    content="",
                    status="error",
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        self.results.extend(processed_results)
        return processed_results
    
    async def crawl_with_seeds(self, seed_urls: List[str], max_depth: int = 2) -> List[CrawlResult]:
        """Crawl starting from seed URLs with link discovery"""
        all_results = []
        urls_to_crawl = seed_urls.copy()
        crawled_urls = set()
        
        for depth in range(max_depth):
            if not urls_to_crawl:
                break
            
            # Crawl current level
            level_results = await self.crawl_urls(urls_to_crawl)
            all_results.extend(level_results)
            
            # Extract new URLs for next level
            new_urls = []
            for result in level_results:
                if result.status == "success":
                    crawled_urls.add(result.url)
                    # Extract links from content (simplified)
                    links = self._extract_links(result.content, result.url)
                    for link in links:
                        if link not in crawled_urls and link not in urls_to_crawl:
                            new_urls.append(link)
            
            urls_to_crawl = new_urls[:self.max_concurrent * 2]  # Limit for next level
        
        return all_results
    
    def _extract_links(self, content: str, base_url: str) -> List[str]:
        """Extract links from content (simplified implementation)"""
        links = []
        # This is a simplified link extraction - in practice you'd use BeautifulSoup
        link_pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(link_pattern, content)
        
        for match in matches:
            if match.startswith('http'):
                links.append(match)
            elif match.startswith('/'):
                links.append(urljoin(base_url, match))
        
        return links
    
    def save_results(self, output_path: str = "data_collection/crawled_data.json") -> bool:
        """Save crawl results to file"""
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert results to serializable format
            serializable_results = []
            for result in self.results:
                serializable_results.append({
                    'url': result.url,
                    'content': result.content,
                    'status': result.status,
                    'error': result.error,
                    'metadata': result.metadata,
                    'timestamp': result.timestamp
                })
            
            data = {
                'crawl_summary': {
                    'total_urls': len(self.results),
                    'successful': len([r for r in self.results if r.status == 'success']),
                    'errors': len([r for r in self.results if r.status == 'error']),
                    'filtered': len([r for r in self.results if r.status == 'filtered']),
                    'timestamp': datetime.now().isoformat()
                },
                'results': serializable_results
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Crawl results saved to {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving crawl results: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get crawling statistics"""
        successful = [r for r in self.results if r.status == 'success']
        errors = [r for r in self.results if r.status == 'error']
        filtered = [r for r in self.results if r.status == 'filtered']
        
        total_content_length = sum(len(r.content) for r in successful)
        
        return {
            'total_urls': len(self.results),
            'successful_crawls': len(successful),
            'errors': len(errors),
            'filtered': len(filtered),
            'total_content_length': total_content_length,
            'average_content_length': total_content_length / len(successful) if successful else 0,
            'domains_crawled': len(self.domain_pages),
            'unique_urls': len(self.visited_urls)
        }

async def main():
    """Example usage of the web crawler"""
    # Example seed URLs for different types of content
    seed_urls = [
        "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "https://github.com/pytorch/pytorch",
        "https://stackoverflow.com/questions/tagged/python",
        "https://docs.python.org/3/tutorial/",
        "https://huggingface.co/docs/transformers/index"
    ]
    
    async with WebCrawler(max_concurrent=5, request_delay=2.0) as crawler:
        print("Starting web crawl...")
        
        # Crawl with link discovery
        results = await crawler.crawl_with_seeds(seed_urls, max_depth=1)
        
        # Print statistics
        stats = crawler.get_statistics()
        print(f"\nCrawl Statistics:")
        print(f"  - Total URLs: {stats['total_urls']}")
        print(f"  - Successful: {stats['successful_crawls']}")
        print(f"  - Errors: {stats['errors']}")
        print(f"  - Filtered: {stats['filtered']}")
        print(f"  - Total Content: {stats['total_content_length']} characters")
        print(f"  - Domains Crawled: {stats['domains_crawled']}")
        
        # Save results
        crawler.save_results()
        
        print("\nCrawl completed successfully!")

if __name__ == "__main__":
    asyncio.run(main())
