import streamlit as st
import pandas as pd
from io import StringIO
import time
from qa_engine import FrenchTransitionQA
from utils import parse_article_file

try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

def get_translations():
    return {
        "fr": {
            "page_title": "QA Tool for French News Transitions",
            "main_title": "📰 QA Tool for French News Transitions", 
            "subtitle": "Analyse la qualité des phrases de transition dans les articles de presse français",
            "loading_models": "Chargement des modèles NLP...",
            "configuration": "Configuration",
            "language": "Langue",
            "similarity_thresholds": "Seuils de similarité",
            "min_threshold_next": "Seuil minimal - transition→paragraphe suivant",
            "max_threshold_prev": "Seuil maximal - transition→paragraphe précédent",
            "min_threshold_help": "Score de similarité minimal requis entre la transition et le paragraphe suivant",
            "max_threshold_help": "Score de similarité maximal autorisé entre la transition et le paragraphe précédent",
            "file_upload": "📁 Téléchargement des fichiers",
            "select_files": "Sélectionnez les fichiers d'articles (.txt)",
            "upload_help": "Téléchargez un ou plusieurs fichiers d'articles au format txt",
            "files_uploaded": "✅ {count} fichier(s) téléchargé(s)",
            "analyze_button": "🔍 Analyser les articles",
            "processing_file": "Traitement de {filename}...",
            "analysis_complete": "Analyse terminée!",
            "no_transitions": "Aucune transition trouvée dans les fichiers analysés.",
            "general_stats": "📊 Statistiques générales",
            "transitions_analyzed": "Transitions analysées",
            "compliance_rate": "Taux de conformité",
            "non_compliant": "Transitions non conformes",
            "avg_similarity": "Similarité moy. (suivant)",
            "failure_analysis": "🔍 Analyse des échecs",
            "failure_types": "Types d'échecs:",
            "failure_details": "Détail des échecs:",
            "repetition_analysis": "🔄 Analyse des répétitions",
            "top_repeated_lemmas": "Top 10 des lemmes les plus répétés:",
            "no_repetitions": "Aucune répétition de lemme détectée.",
            "detailed_results": "📋 Résultats détaillés",
            "status": "Statut",
            "compliant": "✅ Conforme",
            "non_compliant_status": "❌ Non conforme",
            "next_similarity": "Similarité suivant",
            "prev_similarity": "Similarité précédent",
            "export_results": "💾 Export des résultats",
            "download_csv": "📄 Télécharger CSV",
            "download_html": "🌐 Télécharger HTML",
            "detailed_explanations": "📝 Explications détaillées des échecs",
            "all_compliant": "🎉 Toutes les transitions sont conformes!",
            "transition": "Transition:",
            "position": "Position:",
            "failure_reasons": "Raisons des échecs:",
            "triggered_rules": "Règles déclenchées:",
            "next_para_similarity": "Similarité avec paragraphe suivant:",
            "prev_para_similarity": "Similarité avec paragraphe précédent:",
            "repetitions": "répétitions"
        },
        "en": {
            "page_title": "QA Tool for French News Transitions",
            "main_title": "📰 QA Tool for French News Transitions",
            "subtitle": "Analyze the quality of transition phrases in French news articles",
            "loading_models": "Loading NLP models...",
            "configuration": "Configuration",
            "language": "Language",
            "similarity_thresholds": "Similarity Thresholds",
            "min_threshold_next": "Minimum threshold - transition→next paragraph",
            "max_threshold_prev": "Maximum threshold - transition→previous paragraph", 
            "min_threshold_help": "Minimum similarity score required between transition and next paragraph",
            "max_threshold_help": "Maximum similarity score allowed between transition and previous paragraph",
            "file_upload": "📁 File Upload",
            "select_files": "Select article files (.txt)",
            "upload_help": "Upload one or more article files in txt format",
            "files_uploaded": "✅ {count} file(s) uploaded",
            "analyze_button": "🔍 Analyze Articles",
            "processing_file": "Processing {filename}...",
            "analysis_complete": "Analysis complete!",
            "no_transitions": "No transitions found in analyzed files.",
            "general_stats": "📊 General Statistics",
            "transitions_analyzed": "Transitions Analyzed",
            "compliance_rate": "Compliance Rate",
            "non_compliant": "Non-compliant Transitions",
            "avg_similarity": "Avg. Similarity (next)",
            "failure_analysis": "🔍 Failure Analysis",
            "failure_types": "Failure Types:",
            "failure_details": "Failure Details:",
            "repetition_analysis": "🔄 Repetition Analysis",
            "top_repeated_lemmas": "Top 10 Most Repeated Lemmas:",
            "no_repetitions": "No lemma repetitions detected.",
            "detailed_results": "📋 Detailed Results",
            "status": "Status",
            "compliant": "✅ Compliant",
            "non_compliant_status": "❌ Non-compliant",
            "next_similarity": "Next Similarity",
            "prev_similarity": "Prev Similarity", 
            "export_results": "💾 Export Results",
            "download_csv": "📄 Download CSV",
            "download_html": "🌐 Download HTML",
            "detailed_explanations": "📝 Detailed Failure Explanations",
            "all_compliant": "🎉 All transitions are compliant!",
            "transition": "Transition:",
            "position": "Position:",
            "failure_reasons": "Failure Reasons:",
            "triggered_rules": "Triggered Rules:",
            "next_para_similarity": "Similarity with next paragraph:",
            "prev_para_similarity": "Similarity with previous paragraph:",
            "repetitions": "repetitions"
        }
    }

def main():
    st.set_page_config(
        page_title="QA Tool for French News Transitions",
        page_icon="📰",
        layout="wide"
    )
    
    # Get translations
    translations = get_translations()
    
    # Language selector in sidebar
    st.sidebar.header("🌐 Language / Langue")
    language = st.sidebar.selectbox(
        "Select Language / Choisir la langue:",
        ["fr", "en"],
        format_func=lambda x: "🇫🇷 Français" if x == "fr" else "🇺🇸 English"
    )
    
    # Get current translations
    t = translations[language]
    
    st.title(t["main_title"])
    st.markdown(t["subtitle"])
    
    # Initialize QA engine
    @st.cache_resource
    def load_qa_engine():
        with st.spinner(t["loading_models"]):
            return FrenchTransitionQA()
    
    qa_engine = load_qa_engine()
    
    # Sidebar for configuration
    st.sidebar.header(t["configuration"])
    
    # Thresholds
    st.sidebar.subheader(t["similarity_thresholds"])
    similarity_threshold_next = st.sidebar.slider(
        t["min_threshold_next"],
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help=t["min_threshold_help"]
    )
    
    similarity_threshold_prev = st.sidebar.slider(
        t["max_threshold_prev"],
        min_value=0.0,
        max_value=1.0,
        value=0.7,
        step=0.05,
        help=t["max_threshold_help"]
    )
    
    # File upload
    st.header(t["file_upload"])
    uploaded_files = st.file_uploader(
        t["select_files"],
        type=['txt'],
        accept_multiple_files=True,
        help=t["upload_help"]
    )
    
    if uploaded_files:
        st.success(t["files_uploaded"].format(count=len(uploaded_files)))
        
        # Process files button
        if st.button(t["analyze_button"], type="primary"):
            analyze_articles(uploaded_files, qa_engine, similarity_threshold_next, similarity_threshold_prev, t)

def analyze_articles(uploaded_files, qa_engine, similarity_threshold_next, similarity_threshold_prev, t):
    """Process uploaded articles and display results"""
    
    # Progress tracking
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    all_results = []
    total_transitions = 0
    
    # Process each file
    for i, uploaded_file in enumerate(uploaded_files):
        status_text.text(t["processing_file"].format(filename=uploaded_file.name))
        
        try:
            # Read and parse file
            content = StringIO(uploaded_file.getvalue().decode("utf-8")).read()
            article_data = parse_article_file(content, uploaded_file.name)
            
            if article_data:
                # Analyze transitions
                results = qa_engine.analyze_article(
                    article_data,
                    similarity_threshold_next=similarity_threshold_next,
                    similarity_threshold_prev=similarity_threshold_prev
                )
                all_results.extend(results)
                total_transitions += len(results)
            
        except Exception as e:
            error_msg = f"Error processing {uploaded_file.name}: {str(e)}" if t == get_translations()["en"] else f"Erreur lors du traitement de {uploaded_file.name}: {str(e)}"
            st.error(error_msg)
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    status_text.text(t["analysis_complete"])
    time.sleep(0.5)
    status_text.empty()
    progress_bar.empty()
    
    if all_results:
        display_results(all_results, total_transitions, t)
    else:
        st.warning(t["no_transitions"])

def display_results(results, total_transitions, t):
    """Display analysis results"""
    
    # Convert to DataFrame
    df = pd.DataFrame(results)
    
    # Overall statistics
    st.header(t["general_stats"])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(t["transitions_analyzed"], total_transitions)
    
    with col2:
        compliant_count = len(df[df['overall_pass'] == True])
        compliance_rate = (compliant_count / total_transitions * 100) if total_transitions > 0 else 0
        st.metric(t["compliance_rate"], f"{compliance_rate:.1f}%")
    
    with col3:
        failed_count = total_transitions - compliant_count
        st.metric(t["non_compliant"], failed_count)
    
    with col4:
        avg_similarity_next = df['similarity_next'].mean() if 'similarity_next' in df.columns else 0
        st.metric(t["avg_similarity"], f"{avg_similarity_next:.2f}")
    
    # Failure analysis
    st.subheader(t["failure_analysis"])
    
    if failed_count > 0:
        # Count failure types
        failure_reasons = []
        for _, row in df.iterrows():
            if not row['overall_pass']:
                if 'failure_reasons' in row and row['failure_reasons'] is not None and len(row['failure_reasons']) > 0:
                    failure_reasons.extend(row['failure_reasons'])
        
        if failure_reasons:
            failure_df = pd.DataFrame({'reason': failure_reasons})
            failure_counts = failure_df['reason'].value_counts()
            
            col1, col2 = st.columns(2)
            
            with col1:
                if PLOTLY_AVAILABLE:
                    fig_pie = px.pie(
                        values=failure_counts.values,
                        names=failure_counts.index,
                        title="Types d'échecs"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.write(f"**{t['failure_types']}**")
                    for reason, count in failure_counts.items():
                        percentage = (count / sum(failure_counts.values)) * 100
                        st.write(f"• {reason}: {count} ({percentage:.1f}%)")
            
            with col2:
                st.write(f"**{t['failure_details']}**")
                for reason, count in failure_counts.items():
                    st.write(f"• {reason}: {count}")
    
    # Lemma repetition analysis
    st.subheader(t["repetition_analysis"])
    
    # Count repeated lemmas across all articles
    all_repeated_lemmas = []
    for _, row in df.iterrows():
        if 'repeated_lemmas' in row and row['repeated_lemmas'] is not None and len(row['repeated_lemmas']) > 0:
            all_repeated_lemmas.extend(row['repeated_lemmas'])
    
    if all_repeated_lemmas:
        lemma_df = pd.DataFrame({'lemma': all_repeated_lemmas})
        lemma_counts = lemma_df['lemma'].value_counts().head(10)
        
        if PLOTLY_AVAILABLE:
            fig_bar = px.bar(
                x=lemma_counts.values,
                y=lemma_counts.index,
                orientation='h',
                title="Top 10 des lemmes les plus répétés",
                labels={'x': 'Nombre de répétitions', 'y': 'Lemme'}
            )
            fig_bar.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig_bar, use_container_width=True)
        else:
            st.write(f"**{t['top_repeated_lemmas']}**")
            for lemma, count in lemma_counts.items():
                st.write(f"• {lemma}: {count} {t['repetitions']}")
    else:
        st.info(t["no_repetitions"])
    
    # Detailed results table
    st.header(t["detailed_results"])
    
    # Prepare display DataFrame
    display_df = df.copy()
    
    # Format columns for display
    if 'overall_pass' in display_df.columns:
        display_df[t["status"]] = display_df['overall_pass'].apply(lambda x: t["compliant"] if x else t["non_compliant_status"])
    
    if 'similarity_next' in display_df.columns:
        display_df[t["next_similarity"]] = display_df['similarity_next'].round(3)
    
    if 'similarity_prev' in display_df.columns:
        display_df[t["prev_similarity"]] = display_df['similarity_prev'].round(3)
    
    # Select columns to display
    columns_to_show = [
        'article_id', 'para_idx', 'transition_text', t["status"],
        'word_count_pass', 'position_pass', 'repetition_pass', 'cohesion_pass',
        t["next_similarity"], t["prev_similarity"]
    ]
    
    # Filter columns that exist
    available_columns = [col for col in columns_to_show if col in display_df.columns]
    
    # Display table with filtering
    st.dataframe(
        display_df[available_columns],
        use_container_width=True,
        hide_index=True
    )
    
    # Export options
    st.header(t["export_results"])
    
    col1, col2 = st.columns(2)
    
    with col1:
        # CSV export
        csv_data = df.to_csv(index=False)
        st.download_button(
            label=t["download_csv"],
            data=csv_data,
            file_name=f"qa_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    
    with col2:
        # HTML export
        html_data = df.to_html(index=False, escape=False)
        st.download_button(
            label=t["download_html"],
            data=html_data,
            file_name=f"qa_results_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.html",
            mime="text/html"
        )
    
    # Detailed failure explanations
    st.header(t["detailed_explanations"])
    
    failed_transitions = df[df['overall_pass'] == False]
    
    if len(failed_transitions) > 0:
        for _, row in failed_transitions.iterrows():
            with st.expander(f"❌ {row['article_id']} - Paragraphe {row['para_idx']}: \"{row['transition_text']}\""):
                st.write(f"**{t['transition']}** {row['transition_text']}")
                st.write(f"**{t['position']}** Paragraphe {row['para_idx']}")
                
                # Show translation verification if available
                if 'transition_en' in row and row['transition_en'] and not pd.isna(row['transition_en']):
                    if row['transition_en'].startswith('[VERIFY:'):
                        st.warning("🔍 **Vérification manuelle recommandée**: Traduction automatique requise pour validation complète")
                    else:
                        st.info(f"🌐 **Traduction**: {row['transition_en']}")
                
                if 'failure_reasons' in row and row['failure_reasons'] is not None and len(row['failure_reasons']) > 0:
                    st.write(f"**{t['failure_reasons']}**")
                    for reason in row['failure_reasons']:
                        st.write(f"• {reason}")
                
                if 'triggered_rules' in row and row['triggered_rules'] is not None and len(row['triggered_rules']) > 0:
                    st.write(f"**{t['triggered_rules']}**")
                    for rule in row['triggered_rules']:
                        st.write(f"• {rule}")
                
                # Show similarity scores and translation context
                if 'similarity_next' in row:
                    st.write(f"**{t['next_para_similarity']}** {row['similarity_next']:.3f}")
                if 'similarity_prev' in row:
                    st.write(f"**{t['prev_para_similarity']}** {row['similarity_prev']:.3f}")
                
                # Show semantic analysis details if available
                if 'common_concepts' in row and row['common_concepts'] and not pd.isna(row['common_concepts']):
                    try:
                        concepts = eval(row['common_concepts']) if isinstance(row['common_concepts'], str) else row['common_concepts']
                        if concepts:
                            st.write(f"**Concepts communs identifiés**: {', '.join(concepts)}")
                    except:
                        pass
    else:
        st.success(t["all_compliant"])
    
    # Add translation verification summary
    st.header("🔍 Résumé de la vérification par traduction" if t == get_translations()["fr"] else "🔍 Translation Verification Summary")
    
    # Count translations that need manual verification
    verification_needed = 0
    total_analyzed = 0
    
    for _, row in df.iterrows():
        total_analyzed += 1
        if 'transition_en' in row and row['transition_en'] and isinstance(row['transition_en'], str):
            if row['transition_en'].startswith('[VERIFY:'):
                verification_needed += 1
    
    if verification_needed > 0:
        st.warning(f"⚠️ **{verification_needed}** transitions sur **{total_analyzed}** nécessitent une vérification manuelle de traduction pour une analyse complète.")
        st.info("💡 **Recommandation**: Utilisez DeepL ou Google Translate pour vérifier les transitions marquées '[VERIFY:]' afin d'assurer la précision de l'analyse sémantique.")
    else:
        st.success("✅ Toutes les transitions ont été analysées avec vérification automatique par traduction.")

if __name__ == "__main__":
    main()
