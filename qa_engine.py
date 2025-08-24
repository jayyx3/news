try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    import numpy as np

from typing import List, Dict, Any, Tuple
import re
import urllib.parse
import urllib.request
import json

class FrenchTransitionQA:
    """
    QA Engine for analyzing French news transition phrases with translation verification
    """
    
    def __init__(self):
        """Initialize the QA engine with French NLP models"""
        self.nlp = None
        self.sentence_model = None
        self.translation_cache = {}  # Cache translations to avoid redundant API calls
        
        if SPACY_AVAILABLE:
            try:
                # Load French spaCy model
                self.nlp = spacy.load("fr_core_news_md")
            except IOError:
                # Fallback to smaller model
                try:
                    self.nlp = spacy.load("fr_core_news_sm")
                except IOError:
                    print("Warning: No French spaCy model found. Lemma detection will use basic text processing.")
                    self.nlp = None
        
        if TRANSFORMERS_AVAILABLE:
            try:
                self.sentence_model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except Exception:
                # Fallback to a different multilingual model
                try:
                    self.sentence_model = SentenceTransformer('distiluse-base-multilingual-cased')
                except Exception:
                    print("Warning: Cannot load sentence transformer model. Semantic similarity will be basic.")
                    self.sentence_model = None
    
    def analyze_article(self, article_data: Dict[str, Any], 
                       similarity_threshold_next: float = 0.3,
                       similarity_threshold_prev: float = 0.7) -> List[Dict[str, Any]]:
        """
        Analyze all transitions in an article
        
        Args:
            article_data: Dictionary containing article information
            similarity_threshold_next: Minimum similarity threshold for transition->next paragraph
            similarity_threshold_prev: Maximum similarity threshold for transition->previous paragraph
        
        Returns:
            List of analysis results for each transition
        """
        results = []
        
        if not article_data or 'transitions' not in article_data:
            return results
        
        article_id = article_data.get('filename', 'unknown')
        article_text = article_data.get('article_text', '')
        transitions = article_data.get('transitions', [])
        paragraphs = article_data.get('paragraphs', [])
        
        # Get lemmas for the entire article for repetition detection
        article_lemmas = self._extract_article_lemmas(article_text)
        
        # Extract all transition texts for semantic repetition analysis
        all_transition_texts = [t.get('text', '').strip() for t in transitions if t.get('text', '').strip()]
        
        # Perform semantic repetition analysis using translation verification
        semantic_repetition_analysis = self.detect_semantic_repetition(all_transition_texts)
        
        for i, transition_info in enumerate(transitions):
            transition_text = transition_info.get('text', '').strip()
            para_idx = transition_info.get('paragraph_index', i)
            
            if not transition_text:
                continue
            
            # Analyze this transition
            result = self._analyze_single_transition(
                transition_text=transition_text,
                para_idx=para_idx,
                article_id=article_id,
                paragraphs=paragraphs,
                article_lemmas=article_lemmas,
                total_paragraphs=len(paragraphs),
                similarity_threshold_next=similarity_threshold_next,
                similarity_threshold_prev=similarity_threshold_prev
            )
            
            # Add semantic repetition information
            result['semantic_repetition_analysis'] = semantic_repetition_analysis
            
            # Check if this transition is involved in semantic repetitions
            for rep in semantic_repetition_analysis['repetitions']:
                if rep['transition1_idx'] == i or rep['transition2_idx'] == i:
                    if 'failure_reasons' not in result:
                        result['failure_reasons'] = []
                    if 'triggered_rules' not in result:
                        result['triggered_rules'] = []
                    
                    result['failure_reasons'].append(
                        f"Répétition sémantique détectée via traduction (similarité: {rep['similarity_score']:.3f})"
                    )
                    result['triggered_rules'].append("Règle de non-répétition sémantique (vérification par traduction)")
                    result['repetition_pass'] = False
                    result['overall_pass'] = False
            
            results.append(result)
        
        return results
    
    def _analyze_single_transition(self, transition_text: str, para_idx: int, 
                                 article_id: str, paragraphs: List[str],
                                 article_lemmas: List[str], total_paragraphs: int,
                                 similarity_threshold_next: float,
                                 similarity_threshold_prev: float) -> Dict[str, Any]:
        """Analyze a single transition phrase"""
        
        result = {
            'article_id': article_id,
            'para_idx': para_idx,
            'transition_text': transition_text,
            'word_count': 0,
            'word_count_pass': False,
            'position_pass': False,
            'repetition_pass': False,
            'cohesion_pass': False,
            'overall_pass': False,
            'failure_reasons': [],
            'triggered_rules': [],
            'similarity_next': 0.0,
            'similarity_prev': 0.0,
            'repeated_lemmas': []
        }
        
        # 1. Word count check (≤ 5 words)
        word_count_result = self._check_word_count(transition_text)
        result.update(word_count_result)
        
        # 2. Position check (concluding transitions only in final position)
        position_result = self._check_position(transition_text, para_idx, total_paragraphs)
        result.update(position_result)
        
        # 3. Repetition check
        repetition_result = self._check_repetition(transition_text, article_lemmas)
        result.update(repetition_result)
        
        # 4. Thematic cohesion check
        cohesion_result = self._check_thematic_cohesion(
            transition_text, para_idx, paragraphs,
            similarity_threshold_next, similarity_threshold_prev
        )
        result.update(cohesion_result)
        
        # Overall pass/fail
        result['overall_pass'] = (
            result['word_count_pass'] and 
            result['position_pass'] and 
            result['repetition_pass'] and 
            result['cohesion_pass']
        )
        
        return result
    
    def _check_word_count(self, transition_text: str) -> Dict[str, Any]:
        """Check if transition has ≤ 5 words"""
        # Clean and tokenize
        cleaned_text = re.sub(r'[^\w\s]', '', transition_text).strip()
        words = cleaned_text.split()
        word_count = len(words)
        
        is_valid = word_count <= 5
        
        result = {
            'word_count': word_count,
            'word_count_pass': is_valid
        }
        
        if not is_valid:
            if 'failure_reasons' not in result:
                result['failure_reasons'] = []
            if 'triggered_rules' not in result:
                result['triggered_rules'] = []
            result['failure_reasons'].append(f"Nombre de mots dépassé: {word_count}/5")
            result['triggered_rules'].append("Règle des 5 mots maximum")
        
        return result
    
    def _check_position(self, transition_text: str, para_idx: int, total_paragraphs: int) -> Dict[str, Any]:
        """Check position rules for concluding transitions"""
        
        # Define concluding transition patterns
        concluding_patterns = [
            r'\b(pour\s+finir|pour\s+terminer|en\s+conclusion|enfin|finalement)\b',
            r'\b(pour\s+conclure|en\s+guise\s+de\s+conclusion)\b',
            r'\b(dernière?\s+point|dernier\s+élément)\b'
        ]
        
        is_concluding = any(
            re.search(pattern, transition_text.lower()) 
            for pattern in concluding_patterns
        )
        
        # If it's a concluding transition, it should be in the final position
        is_final_position = para_idx >= total_paragraphs - 1
        
        is_valid = not is_concluding or is_final_position
        
        result = {'position_pass': is_valid}
        
        if not is_valid:
            if 'failure_reasons' not in result:
                result['failure_reasons'] = []
            if 'triggered_rules' not in result:
                result['triggered_rules'] = []
            result['failure_reasons'].append(
                f"Transition de conclusion en position {para_idx + 1} sur {total_paragraphs} paragraphes"
            )
            result['triggered_rules'].append("Règle de placement des transitions de conclusion")
        
        return result
    
    def _check_repetition(self, transition_text: str, article_lemmas: List[str]) -> Dict[str, Any]:
        """Check for lemma repetition in transition"""
        
        if self.nlp:
            # Extract lemmas from transition using spaCy
            doc = self.nlp(transition_text)
            transition_lemmas = [token.lemma_.lower() for token in doc 
                               if not token.is_stop and not token.is_punct and token.is_alpha]
        else:
            # Fallback: use basic word processing
            transition_words = re.findall(r'\b\w+\b', transition_text.lower())
            transition_lemmas = [word for word in transition_words if len(word) > 2]
        
        # Find repeated lemmas
        repeated_lemmas = []
        for lemma in transition_lemmas:
            if article_lemmas.count(lemma) > 1:  # Appears more than once in article
                repeated_lemmas.append(lemma)
        
        is_valid = len(repeated_lemmas) == 0
        
        result = {
            'repetition_pass': is_valid,
            'repeated_lemmas': repeated_lemmas
        }
        
        if not is_valid:
            result['failure_reasons'] = [
                f"Répétition de lemmes détectée: {', '.join(repeated_lemmas)}"
            ]
            result['triggered_rules'] = ["Règle de non-répétition des lemmes"]
        
        return result
    
    def _check_thematic_cohesion(self, transition_text: str, para_idx: int, 
                                paragraphs: List[str], similarity_threshold_next: float,
                                similarity_threshold_prev: float) -> Dict[str, Any]:
        """Check thematic cohesion using semantic similarity"""
        
        result = {
            'cohesion_pass': True,
            'similarity_next': 0.0,
            'similarity_prev': 0.0
        }
        
        failure_reasons = []
        triggered_rules = []
        
        if self.sentence_model:
            try:
                # Get embeddings for transition
                transition_embedding = self.sentence_model.encode([transition_text])
                
                # Check similarity with next paragraph
                if para_idx + 1 < len(paragraphs):
                    next_para = paragraphs[para_idx + 1]
                    next_embedding = self.sentence_model.encode([next_para])
                    similarity_next = float(np.dot(transition_embedding[0], next_embedding[0]) / 
                                          (np.linalg.norm(transition_embedding[0]) * np.linalg.norm(next_embedding[0])))
                    result['similarity_next'] = similarity_next
                    
                    if similarity_next < similarity_threshold_next:
                        result['cohesion_pass'] = False
                        failure_reasons.append(
                            f"Similarité trop faible avec paragraphe suivant: {similarity_next:.3f} < {similarity_threshold_next}"
                        )
                        triggered_rules.append("Règle de cohésion thématique (paragraphe suivant)")
                
                # Check similarity with previous paragraph
                if para_idx > 0:
                    prev_para = paragraphs[para_idx - 1]
                    prev_embedding = self.sentence_model.encode([prev_para])
                    similarity_prev = float(np.dot(transition_embedding[0], prev_embedding[0]) / 
                                          (np.linalg.norm(transition_embedding[0]) * np.linalg.norm(prev_embedding[0])))
                    result['similarity_prev'] = similarity_prev
                    
                    if similarity_prev > similarity_threshold_prev:
                        result['cohesion_pass'] = False
                        failure_reasons.append(
                            f"Similarité trop élevée avec paragraphe précédent: {similarity_prev:.3f} > {similarity_threshold_prev}"
                        )
                        triggered_rules.append("Règle de cohésion thématique (paragraphe précédent)")
            
            except Exception as e:
                result['cohesion_pass'] = False
                failure_reasons.append(f"Erreur lors du calcul de similarité: {str(e)}")
                triggered_rules.append("Erreur de calcul de cohésion")
        else:
            # Fallback: use translation-based verification
            if para_idx + 1 < len(paragraphs):
                coherence_check = self.verify_thematic_coherence(transition_text, paragraphs[para_idx + 1])
                result['similarity_next'] = coherence_check['semantic_overlap']
                result['transition_en'] = coherence_check['transition_en']
                result['next_paragraph_en'] = coherence_check['paragraph_en']
                result['common_concepts'] = coherence_check['common_concepts']
                
                if coherence_check['semantic_overlap'] < 0.1:  # Basic threshold
                    result['cohesion_pass'] = False
                    failure_reasons.append(f"Faible cohérence thématique détectée via analyse traduite: {coherence_check['semantic_overlap']:.3f}")
                    triggered_rules.append("Règle de cohésion thématique (vérification par traduction)")
                    
                    # Add translation context for debugging
                    if coherence_check['transition_en'].startswith('[VERIFY:'):
                        failure_reasons.append(f"Traduction requise pour vérification: {transition_text}")
                        triggered_rules.append("Vérification manuelle recommandée")
        
        if failure_reasons:
            result['failure_reasons'] = failure_reasons
            result['triggered_rules'] = triggered_rules
        
        return result
    
    def _extract_article_lemmas(self, article_text: str) -> List[str]:
        """Extract all lemmas from the article text"""
        if self.nlp:
            doc = self.nlp(article_text)
            lemmas = [token.lemma_.lower() for token in doc 
                     if not token.is_stop and not token.is_punct and token.is_alpha]
            return lemmas
        else:
            # Fallback: use basic word processing
            words = re.findall(r'\b\w+\b', article_text.lower())
            return [word for word in words if len(word) > 2]
    
    def translate_text(self, text: str, target_lang: str = "en") -> str:
        """
        Translate French text to English using a simple translation service
        This is a basic implementation - in production, use DeepL or Google Translate API
        """
        # Check cache first
        cache_key = f"{text}_{target_lang}"
        if cache_key in self.translation_cache:
            return self.translation_cache[cache_key]
        
        # Simple fallback translation for common transition patterns
        translation_dict = {
            "dans l'actualité culturelle": "in cultural news",
            "ce projet original attire": "this original project attracts",
            "pour finir, on annonce que": "finally, we announce that", 
            "dans un tout autre registre": "in a completely different register",
            "une idée originale à tester": "an original idea to test",
            "pour terminer cette revue": "to finish this review",
            "dans un tout autre domaine": "in a completely different field",
            "des nouvelles sportives également": "sports news as well",
            "dans le registre culturel": "in the cultural register",
            "signalons aussi un événement marquant": "let's also note a significant event",
            "dernier point à noter": "last point to note",
            "côté sportif, on annonce que": "on the sports side, we announce that",
            "en guise de conclusion": "as a conclusion",
            "au sujet des infrastructures": "regarding infrastructure",
            "enfin, et pour finir": "finally, and to finish",
            "côté associations, on note que": "regarding associations, we note that",
            "enfin, pour finir, notons que": "finally, to finish, let's note that",
            "partons à présent à la découverte": "let's now discover",
            "pour conclure cette sélection": "to conclude this selection",
            "enfin, sachez que": "finally, know that",
            "préparez-vous à vibrer au rythme de la musique": "get ready to vibrate to the rhythm of music"
        }
        
        text_lower = text.lower().strip()
        if text_lower in translation_dict:
            translated = translation_dict[text_lower]
        else:
            # For unknown phrases, return a marker for manual verification
            translated = f"[VERIFY: {text}]"
        
        # Cache the result
        self.translation_cache[cache_key] = translated
        return translated
    
    def verify_thematic_coherence(self, transition_text: str, paragraph_text: str) -> Dict[str, Any]:
        """
        Verify thematic coherence between transition and paragraph using translation
        """
        # Translate both texts for semantic analysis
        transition_en = self.translate_text(transition_text)
        paragraph_en = self.translate_text(paragraph_text[:200])  # First 200 chars for context
        
        # Basic keyword overlap analysis on English text
        transition_words = set(re.findall(r'\b\w+\b', transition_en.lower()))
        paragraph_words = set(re.findall(r'\b\w+\b', paragraph_en.lower()))
        
        # Remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'between', 'among', 'within', 'without', 'under', 'over', 'across', 'around', 'near', 'against', 'along', 'beside', 'beyond', 'inside', 'outside', 'throughout', 'underneath', 'upon', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'can', 'shall'}
        
        transition_words = transition_words - stop_words
        paragraph_words = paragraph_words - stop_words
        
        # Calculate semantic overlap
        common_words = transition_words & paragraph_words
        overlap_score = len(common_words) / max(len(transition_words), 1) if transition_words else 0
        
        return {
            'transition_en': transition_en,
            'paragraph_en': paragraph_en,
            'semantic_overlap': overlap_score,
            'common_concepts': list(common_words)
        }
    
    def detect_semantic_repetition(self, transitions: List[str]) -> Dict[str, Any]:
        """
        Detect semantic repetition between transitions using translation verification
        """
        if len(transitions) < 2:
            return {'repetitions': [], 'analysis': {}}
        
        # Translate all transitions
        translated = [self.translate_text(t) for t in transitions]
        
        repetitions = []
        analysis = {}
        
        for i, trans1 in enumerate(translated):
            for j, trans2 in enumerate(translated[i+1:], i+1):
                # Skip verification markers
                if trans1.startswith('[VERIFY:') or trans2.startswith('[VERIFY:'):
                    continue
                
                # Basic similarity check on English translations
                words1 = set(re.findall(r'\b\w+\b', trans1.lower()))
                words2 = set(re.findall(r'\b\w+\b', trans2.lower()))
                
                if len(words1) > 0 and len(words2) > 0:
                    overlap = len(words1 & words2) / min(len(words1), len(words2))
                    if overlap > 0.6:  # High similarity threshold
                        repetitions.append({
                            'transition1_idx': i,
                            'transition2_idx': j, 
                            'transition1_fr': transitions[i],
                            'transition2_fr': transitions[j],
                            'transition1_en': trans1,
                            'transition2_en': trans2,
                            'similarity_score': overlap
                        })
        
        analysis['total_transitions'] = len(transitions)
        analysis['repetition_count'] = len(repetitions)
        analysis['translations'] = list(zip(transitions, translated))
        
        return {'repetitions': repetitions, 'analysis': analysis}
