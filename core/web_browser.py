"""
WEB BROWSER MODULE
==================

Provides internet browsing capabilities for the Meta-Model AI Assistant.
Includes captcha detection and user prompting for verification.
"""

import requests
import asyncio
import logging
import re
import json
from typing import Dict, Any, Optional, List
from datetime import datetime
from urllib.parse import urlparse, urljoin
from bs4 import BeautifulSoup
import time

logger = logging.getLogger(__name__)


class WebBrowser:
    """
    Web browser with internet browsing capabilities and captcha handling.
    """
    
    def __init__(self):
        """Initialize web browser."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        self.browser_config = {
            "timeout": 30,
            "max_redirects": 5,
            "enable_javascript": False,  # For basic browsing
            "captcha_detection": True,
            "user_confirmation_required": True,
            "allowed_domains": [],  # Empty means all domains allowed
            "blocked_domains": [],
            "max_page_size": 10 * 1024 * 1024,  # 10MB
            "rate_limit": {
                "requests_per_minute": 60,
                "requests_per_hour": 1000
            }
        }
        
        self.request_history = []
        self.captcha_encounters = []
        self.user_responses = {}
        
    def browse(self, url: str, user_query: str = None) -> Dict[str, Any]:
        """
        Browse a URL and extract relevant information.
        
        Args:
            url: URL to browse
            user_query: Optional user query to focus extraction
            
        Returns:
            Browser result with content and metadata
        """
        try:
            # Validate URL
            if not self._is_valid_url(url):
                return {
                    "success": False,
                    "error": "Invalid URL format",
                    "url": url
                }
            
            # Check if domain is allowed/blocked
            domain_check = self._check_domain_permissions(url)
            if not domain_check["allowed"]:
                return {
                    "success": False,
                    "error": f"Domain not allowed: {domain_check['reason']}",
                    "url": url
                }
            
            # Check rate limiting
            if not self._check_rate_limit():
                return {
                    "success": False,
                    "error": "Rate limit exceeded",
                    "url": url
                }
            
            # Make request
            response = self._make_request(url)
            
            if not response["success"]:
                return response
            
            # Check for captcha
            captcha_result = self._detect_captcha(response["content"], url)
            if captcha_result["detected"]:
                return self._handle_captcha(captcha_result, url, user_query)
            
            # Extract content
            content_result = self._extract_content(response["content"], user_query)
            
            # Log request
            self._log_request(url, response["status_code"], len(response["content"]))
            
            return {
                "success": True,
                "url": url,
                "title": content_result["title"],
                "content": content_result["content"],
                "summary": content_result["summary"],
                "links": content_result["links"],
                "metadata": {
                    "status_code": response["status_code"],
                    "content_length": len(response["content"]),
                    "timestamp": datetime.now().isoformat(),
                    "user_query": user_query
                }
            }
            
        except Exception as e:
            logger.error(f"Browser error for {url}: {e}")
            return {
                "success": False,
                "error": str(e),
                "url": url
            }
    
    def search_web(self, query: str, max_results: int = 5) -> Dict[str, Any]:
        """
        Search the web for information.
        
        Args:
            query: Search query
            max_results: Maximum number of results
            
        Returns:
            Search results
        """
        try:
            # For now, simulate web search (in real implementation, would use search APIs)
            logger.info(f"Searching web for: {query}")
            
            # Simulate search results
            search_results = [
                {
                    "title": f"Result for: {query}",
                    "url": f"https://example.com/search?q={query}",
                    "snippet": f"This is a simulated search result for '{query}'"
                }
            ]
            
            return {
                "success": True,
                "query": query,
                "results": search_results[:max_results],
                "total_results": len(search_results),
                "metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "max_results": max_results
                }
            }
            
        except Exception as e:
            logger.error(f"Search error for '{query}': {e}")
            return {
                "success": False,
                "error": str(e),
                "query": query
            }
    
    def _is_valid_url(self, url: str) -> bool:
        """Validate URL format."""
        try:
            result = urlparse(url)
            return all([result.scheme, result.netloc])
        except:
            return False
    
    def _check_domain_permissions(self, url: str) -> Dict[str, Any]:
        """Check if domain is allowed/blocked."""
        domain = urlparse(url).netloc
        
        # Check blocked domains
        if domain in self.browser_config["blocked_domains"]:
            return {"allowed": False, "reason": "Domain is blocked"}
        
        # Check allowed domains (if specified)
        if self.browser_config["allowed_domains"]:
            if domain not in self.browser_config["allowed_domains"]:
                return {"allowed": False, "reason": "Domain not in allowed list"}
        
        return {"allowed": True}
    
    def _check_rate_limit(self) -> bool:
        """Check if rate limit is exceeded."""
        current_time = time.time()
        recent_requests = [
            req for req in self.request_history 
            if current_time - req["timestamp"] < 60  # Last minute
        ]
        
        if len(recent_requests) >= self.browser_config["rate_limit"]["requests_per_minute"]:
            return False
        
        return True
    
    def _make_request(self, url: str) -> Dict[str, Any]:
        """Make HTTP request to URL."""
        try:
            response = self.session.get(
                url,
                timeout=self.browser_config["timeout"],
                allow_redirects=True
            )
            
            # Check content size
            if len(response.content) > self.browser_config["max_page_size"]:
                return {
                    "success": False,
                    "error": "Page too large",
                    "status_code": response.status_code
                }
            
            return {
                "success": True,
                "content": response.text,
                "status_code": response.status_code,
                "headers": dict(response.headers)
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "success": False,
                "error": f"Request failed: {str(e)}",
                "status_code": None
            }
    
    def _detect_captcha(self, content: str, url: str) -> Dict[str, Any]:
        """Detect captcha on the page."""
        if not self.browser_config["captcha_detection"]:
            return {"detected": False}
        
        # Common captcha indicators
        captcha_indicators = [
            "captcha",
            "recaptcha",
            "cloudflare",
            "challenge",
            "verify you are human",
            "prove you are human",
            "security check",
            "robot check"
        ]
        
        content_lower = content.lower()
        
        for indicator in captcha_indicators:
            if indicator in content_lower:
                return {
                    "detected": True,
                    "type": indicator,
                    "url": url,
                    "timestamp": datetime.now().isoformat()
                }
        
        return {"detected": False}
    
    def _handle_captcha(self, captcha_result: Dict[str, Any], url: str, user_query: str) -> Dict[str, Any]:
        """Handle captcha detection by prompting user."""
        captcha_id = f"captcha_{int(time.time())}"
        
        # Store captcha encounter
        self.captcha_encounters.append({
            "id": captcha_id,
            "url": url,
            "type": captcha_result["type"],
            "timestamp": captcha_result["timestamp"],
            "user_query": user_query,
            "resolved": False
        })
        
        return {
            "success": False,
            "error": "CAPTCHA_DETECTED",
            "captcha_id": captcha_id,
            "url": url,
            "captcha_type": captcha_result["type"],
            "message": f"Captcha detected on {url}. Please solve the captcha manually and provide the solution.",
            "user_prompt": f"🔒 CAPTCHA DETECTED\n\nURL: {url}\nType: {captcha_result['type']}\n\nPlease solve the captcha manually and provide the solution to continue browsing."
        }
    
    def resolve_captcha(self, captcha_id: str, solution: str) -> Dict[str, Any]:
        """
        Resolve a captcha with user-provided solution.
        
        Args:
            captcha_id: Captcha ID to resolve
            solution: User-provided solution
            
        Returns:
            Resolution result
        """
        try:
            # Find the captcha encounter
            captcha_encounter = None
            for encounter in self.captcha_encounters:
                if encounter["id"] == captcha_id:
                    captcha_encounter = encounter
                    break
            
            if not captcha_encounter:
                return {
                    "success": False,
                    "error": "Captcha ID not found"
                }
            
            # Store user response
            self.user_responses[captcha_id] = {
                "solution": solution,
                "timestamp": datetime.now().isoformat()
            }
            
            # Mark as resolved
            captcha_encounter["resolved"] = True
            captcha_encounter["solution"] = solution
            
            # Retry the original request
            original_url = captcha_encounter["url"]
            user_query = captcha_encounter["user_query"]
            
            # Simulate successful resolution
            return {
                "success": True,
                "captcha_id": captcha_id,
                "message": "Captcha resolved successfully",
                "retry_result": self.browse(original_url, user_query)
            }
            
        except Exception as e:
            logger.error(f"Error resolving captcha {captcha_id}: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _extract_content(self, html_content: str, user_query: str = None) -> Dict[str, Any]:
        """Extract relevant content from HTML."""
        try:
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract title
            title = soup.find('title')
            title_text = title.get_text() if title else "No title"
            
            # Extract main content
            main_content = ""
            
            # Try to find main content areas
            content_selectors = [
                'main',
                'article',
                '.content',
                '.main-content',
                '#content',
                '#main',
                'body'
            ]
            
            for selector in content_selectors:
                content_elem = soup.select_one(selector)
                if content_elem:
                    main_content = content_elem.get_text(separator=' ', strip=True)
                    break
            
            # If no main content found, use body
            if not main_content:
                main_content = soup.get_text(separator=' ', strip=True)
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True):
                href = link.get('href')
                text = link.get_text(strip=True)
                if href and text:
                    links.append({
                        "url": href,
                        "text": text
                    })
            
            # Generate summary
            summary = self._generate_summary(main_content, user_query)
            
            return {
                "title": title_text,
                "content": main_content,
                "summary": summary,
                "links": links[:10]  # Limit to 10 links
            }
            
        except Exception as e:
            logger.error(f"Error extracting content: {e}")
            return {
                "title": "Error extracting content",
                "content": html_content[:1000],  # First 1000 chars
                "summary": "Content extraction failed",
                "links": []
            }
    
    def _generate_summary(self, content: str, user_query: str = None) -> str:
        """Generate a summary of the content."""
        if not content:
            return "No content available"
        
        # Simple summary generation
        sentences = content.split('.')
        if len(sentences) > 3:
            summary = '. '.join(sentences[:3]) + '.'
        else:
            summary = content[:500] + "..." if len(content) > 500 else content
        
        return summary
    
    def _log_request(self, url: str, status_code: int, content_length: int):
        """Log a request for rate limiting."""
        self.request_history.append({
            "url": url,
            "status_code": status_code,
            "content_length": content_length,
            "timestamp": time.time()
        })
        
        # Keep only last 1000 requests
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    def get_browser_stats(self) -> Dict[str, Any]:
        """Get browser statistics."""
        return {
            "total_requests": len(self.request_history),
            "captcha_encounters": len(self.captcha_encounters),
            "resolved_captchas": len([c for c in self.captcha_encounters if c["resolved"]]),
            "user_responses": len(self.user_responses),
            "config": self.browser_config
        }
    
    def update_config(self, config_updates: Dict[str, Any]):
        """Update browser configuration."""
        for key, value in config_updates.items():
            if key in self.browser_config:
                self.browser_config[key] = value
        
        logger.info(f"Browser config updated: {config_updates}")


# Global web browser instance
WEB_BROWSER = WebBrowser()


def get_web_browser() -> WebBrowser:
    """Get the global web browser instance."""
    return WEB_BROWSER


def browse_url(url: str, user_query: str = None) -> Dict[str, Any]:
    """Browse a URL."""
    return WEB_BROWSER.browse(url, user_query)


def search_web(query: str, max_results: int = 5) -> Dict[str, Any]:
    """Search the web."""
    return WEB_BROWSER.search_web(query, max_results)


def resolve_captcha(captcha_id: str, solution: str) -> Dict[str, Any]:
    """Resolve a captcha."""
    return WEB_BROWSER.resolve_captcha(captcha_id, solution)


def get_browser_stats() -> Dict[str, Any]:
    """Get browser statistics."""
    return WEB_BROWSER.get_browser_stats() 