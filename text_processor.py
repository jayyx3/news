import re
from typing import List, Dict, Any, Optional

class FrenchTextProcessor:
    """
    Text processor for French news articles
    """
    
    def __init__(self):
        pass
    
    def extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract main sections from article text
        
        Args:
            text: Raw article text
            
        Returns:
            Dictionary with extracted sections
        """
        sections = {
            'title': '',
            'subtitle': '',
            'article': '',
            'transitions': []
        }
        
        lines = text.strip().split('\n')
        current_section = None
        article_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Detect section headers
            if line.startswith('Titre:'):
                sections['title'] = line.replace('Titre:', '').strip()
                current_section = 'title'
            elif line.startswith('Chapeau:'):
                sections['subtitle'] = line.replace('Chapeau:', '').strip()
                current_section = 'subtitle'
            elif line.startswith('Article:'):
                current_section = 'article'
            elif line.startswith('Transitions générées:'):
                current_section = 'transitions'
            elif current_section == 'article':
                article_lines.append(line)
            elif current_section == 'transitions':
                # Extract transition text (remove numbering)
                if re.match(r'^\d+\.\s*', line):
                    transition_text = re.sub(r'^\d+\.\s*', '', line).strip()
                    if transition_text:
                        sections['transitions'].append(transition_text)
        
        sections['article'] = '\n'.join(article_lines)
        return sections
    
    def split_paragraphs(self, article_text: str) -> List[str]:
        """
        Split article text into paragraphs
        
        Args:
            article_text: Article content
            
        Returns:
            List of paragraph texts
        """
        if not article_text:
            return []
        
        # Split by double newlines or single newlines followed by capital letters
        paragraphs = re.split(r'\n\s*\n|\n(?=[A-ZÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖØÙÚÛÜÝÞ])', article_text)
        
        # Clean and filter paragraphs
        cleaned_paragraphs = []
        for para in paragraphs:
            para = para.strip()
            if para and len(para) > 10:  # Filter out very short paragraphs
                cleaned_paragraphs.append(para)
        
        return cleaned_paragraphs
    
    def identify_transition_positions(self, paragraphs: List[str], transitions: List[str]) -> List[Dict[str, Any]]:
        """
        Identify where transitions appear in the article structure
        
        Args:
            paragraphs: List of paragraph texts
            transitions: List of transition texts
            
        Returns:
            List of transition information with positions
        """
        transition_info = []
        
        for i, transition in enumerate(transitions):
            # Try to find the transition in the paragraphs
            found_position = None
            
            for j, paragraph in enumerate(paragraphs):
                # Check if transition appears at the start of a paragraph
                if paragraph.strip().startswith(transition.strip()):
                    found_position = j
                    break
                # Check if transition appears as a standalone paragraph
                elif paragraph.strip() == transition.strip():
                    found_position = j
                    break
            
            # If not found, estimate position based on order
            if found_position is None:
                # Distribute transitions evenly across paragraphs
                estimated_position = min(
                    int((i + 1) * len(paragraphs) / (len(transitions) + 1)),
                    len(paragraphs) - 1
                )
                found_position = estimated_position
            
            transition_info.append({
                'text': transition,
                'paragraph_index': found_position,
                'order': i + 1
            })
        
        return transition_info
    
    def clean_text(self, text: str) -> str:
        """
        Clean text for processing
        
        Args:
            text: Raw text
            
        Returns:
            Cleaned text
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters that might interfere with processing
        text = re.sub(r'[^\w\s\.,;:!?\'"()[\]{}«»""''-]', ' ', text)
        
        return text.strip()
    
    def extract_concluding_phrases(self, text: str) -> List[str]:
        """
        Extract phrases that indicate conclusion
        
        Args:
            text: Text to analyze
            
        Returns:
            List of concluding phrases found
        """
        concluding_patterns = [
            r'pour\s+finir',
            r'pour\s+terminer',
            r'pour\s+conclure',
            r'en\s+conclusion',
            r'en\s+guise\s+de\s+conclusion',
            r'enfin',
            r'finalement',
            r'dernière?\s+point',
            r'dernier\s+élément'
        ]
        
        found_phrases = []
        text_lower = text.lower()
        
        for pattern in concluding_patterns:
            matches = re.findall(pattern, text_lower)
            found_phrases.extend(matches)
        
        return found_phrases
