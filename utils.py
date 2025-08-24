import re
import pandas as pd
from typing import Dict, Any, List, Optional
from text_processor import FrenchTextProcessor

def parse_article_file(content: str, filename: str) -> Optional[Dict[str, Any]]:
    """
    Parse an article file and extract structured data
    
    Args:
        content: File content as string
        filename: Name of the file
        
    Returns:
        Structured article data or None if parsing fails
    """
    try:
        processor = FrenchTextProcessor()
        
        # Extract sections
        sections = processor.extract_sections(content)
        
        if not sections['article'] or not sections['transitions']:
            return None
        
        # Split article into paragraphs
        paragraphs = processor.split_paragraphs(sections['article'])
        
        if not paragraphs:
            return None
        
        # Identify transition positions
        transition_info = processor.identify_transition_positions(
            paragraphs, sections['transitions']
        )
        
        # Create structured data
        article_data = {
            'filename': filename,
            'title': sections['title'],
            'subtitle': sections['subtitle'],
            'article_text': sections['article'],
            'paragraphs': paragraphs,
            'transitions': transition_info,
            'total_paragraphs': len(paragraphs),
            'total_transitions': len(transition_info)
        }
        
        return article_data
    
    except Exception as e:
        print(f"Error parsing {filename}: {str(e)}")
        return None

def validate_article_structure(article_data: Dict[str, Any]) -> List[str]:
    """
    Validate article structure and return any issues found
    
    Args:
        article_data: Structured article data
        
    Returns:
        List of validation issues (empty if valid)
    """
    issues = []
    
    if not article_data:
        issues.append("Article data is None")
        return issues
    
    # Check required fields
    required_fields = ['filename', 'article_text', 'paragraphs', 'transitions']
    for field in required_fields:
        if field not in article_data:
            issues.append(f"Missing required field: {field}")
    
    # Check if article has content
    if 'article_text' in article_data and not article_data['article_text'].strip():
        issues.append("Article text is empty")
    
    # Check if paragraphs exist
    if 'paragraphs' in article_data and len(article_data['paragraphs']) == 0:
        issues.append("No paragraphs found")
    
    # Check if transitions exist
    if 'transitions' in article_data and len(article_data['transitions']) == 0:
        issues.append("No transitions found")
    
    # Check transition structure
    if 'transitions' in article_data:
        for i, transition in enumerate(article_data['transitions']):
            if not isinstance(transition, dict):
                issues.append(f"Transition {i+1} is not a dictionary")
            elif 'text' not in transition:
                issues.append(f"Transition {i+1} missing text field")
            elif not transition['text'].strip():
                issues.append(f"Transition {i+1} has empty text")
    
    return issues

def format_similarity_score(score: float) -> str:
    """
    Format similarity score for display
    
    Args:
        score: Similarity score (0-1)
        
    Returns:
        Formatted score string
    """
    if score < 0.3:
        return f"{score:.3f} (Faible)"
    elif score < 0.7:
        return f"{score:.3f} (Modérée)"
    else:
        return f"{score:.3f} (Élevée)"

def get_rule_description(rule_name: str) -> str:
    """
    Get human-readable description of a rule
    
    Args:
        rule_name: Name of the rule
        
    Returns:
        Description of the rule
    """
    descriptions = {
        "Règle des 5 mots maximum": "Les transitions doivent contenir au maximum 5 mots",
        "Règle de placement des transitions de conclusion": "Les transitions de conclusion ne peuvent apparaître qu'en position finale",
        "Règle de non-répétition des lemmes": "Les lemmes utilisés dans les transitions ne doivent pas se répéter dans l'article",
        "Règle de cohésion thématique (paragraphe suivant)": "La transition doit avoir une similarité suffisante avec le paragraphe suivant",
        "Règle de cohésion thématique (paragraphe précédent)": "La transition ne doit pas être trop similaire au paragraphe précédent"
    }
    
    return descriptions.get(rule_name, rule_name)

def extract_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Extract summary statistics from analysis results
    
    Args:
        results: List of analysis results
        
    Returns:
        Summary statistics
    """
    if not results:
        return {}
    
    total_transitions = len(results)
    compliant_transitions = sum(1 for r in results if r.get('overall_pass', False))
    
    # Count failure types
    failure_counts = {}
    all_repeated_lemmas = []
    
    for result in results:
        if not result.get('overall_pass', True):
            failure_reasons = result.get('failure_reasons', [])
            for reason in failure_reasons:
                failure_counts[reason] = failure_counts.get(reason, 0) + 1
        
        repeated_lemmas = result.get('repeated_lemmas', [])
        all_repeated_lemmas.extend(repeated_lemmas)
    
    # Calculate average similarities
    similarities_next = [r.get('similarity_next', 0) for r in results if 'similarity_next' in r]
    similarities_prev = [r.get('similarity_prev', 0) for r in results if 'similarity_prev' in r]
    
    stats = {
        'total_transitions': total_transitions,
        'compliant_transitions': compliant_transitions,
        'compliance_rate': (compliant_transitions / total_transitions * 100) if total_transitions > 0 else 0,
        'failure_counts': failure_counts,
        'repeated_lemmas': all_repeated_lemmas,
        'avg_similarity_next': sum(similarities_next) / len(similarities_next) if similarities_next else 0,
        'avg_similarity_prev': sum(similarities_prev) / len(similarities_prev) if similarities_prev else 0
    }
    
    return stats

def create_export_data(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Prepare data for export
    
    Args:
        results: Analysis results
        
    Returns:
        Export-ready data
    """
    export_data = {
        'summary': extract_statistics(results),
        'detailed_results': results,
        'metadata': {
            'export_timestamp': str(pd.Timestamp.now()),
            'total_articles': len(set(r.get('article_id', '') for r in results)),
            'analysis_version': '1.0'
        }
    }
    
    return export_data
