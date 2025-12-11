"""
Streamlit UI for Regulatory Compliance RAG System
Simple function-based implementation with Gemini
"""

import streamlit as st
import json
import os
from rag_query_system import initialize_system, query
from narrative_generator import load_llm, generate_narrative
from test_rag_system import load_test_questions, run_test_suite, print_report

# Page configuration
st.set_page_config(
    page_title="Regulatory Compliance RAG System",
    page_icon="",
    layout="wide"
)

st.markdown(
    """
    <style>
    .main {
        background-color: #f6f8fb;
        font-family: 'Inter', sans-serif;
    }
    .app-header {
        background: linear-gradient(90deg, #0f62fe, #3a7bd5);
        padding: 1.2rem 2rem;
        border-radius: 14px;
        color: white;
        margin-bottom: 1.2rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0px 6px 16px rgba(15, 41, 77, 0.08);
    }
    .stTextInput>div>div>input, .stTextArea textarea {
        border-radius: 10px;
    }
    .stButton>button {
        background-color: #0f62fe;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 1rem;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="app-header">
        <h1 style="margin-bottom:0.3rem;">Regulatory Compliance Intelligence</h1>
        <p style="margin:0;">Bring audit-ready policy insights to frontline teams using a modern RAG stack powered by Gemini.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Use the configuration panel to authenticate with Gemini, load the vector store, and tailor responses. Each tab below focuses on a key workflow: policy search, incident narratives, and regression testing."
)

# Initialize session state
if 'system_initialized' not in st.session_state:
    st.session_state.system_initialized = False
if 'llm_model' not in st.session_state:
    st.session_state.llm_model = 'gemini-pro'

# Sidebar
with st.sidebar:
    st.title("⚙️ Configuration")
    
    st.subheader("Google Gemini API")
    api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value="<Your API Key>",
        help="Enter your Google Gemini API key"
    )
    if api_key:
        os.environ['GOOGLE_API_KEY'] = api_key
    
    model_name = st.selectbox(
        "Gemini Model",
        ["gemini-2.5-flash", "gemini-2.5-pro", "gemini-pro", "gemini-flash"],
        index=0
    )
    st.session_state.llm_model = model_name
    
    st.subheader("System Status")
    
    # Check if vector store exists
    vector_store_exists = os.path.exists('vector_store/faiss_index.bin')
    
    if vector_store_exists:
        st.success("✓ Vector Store Available")
    else:
        st.error("✗ Vector Store Not Found")
        st.info("Run rag_pipeline.py first to create the vector store")
    
    # Initialize systems
    if st.button("Initialize Systems") and vector_store_exists:
        with st.spinner("Loading RAG system..."):
            try:
                initialize_system(llm_model=st.session_state.llm_model, api_key=api_key if api_key else None)
                load_llm(st.session_state.llm_model, api_key if api_key else None)
                st.session_state.system_initialized = True
                st.success("✓ Systems initialized!")
            except Exception as e:
                st.error(f"Error: {e}")

# Main content
st.title("Regulatory Compliance RAG System")
st.markdown(
    """
    <div class="app-intro" style="margin-bottom:1.5rem;">
        <p style="font-size:1rem; color:#475467;">
            Deploy a production-ready RAG workflow to search complex policy documents, synthesize guidance with Gemini, and produce auditable narratives for compliance teams.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    " Query System", 
    " Narrative Generator", 
    " Testing", 
    " Status"
])

# Tab 1: Query System
with tab1:
    st.header("Query Regulatory Documents")
    
    if not st.session_state.system_initialized:
        st.warning("Please initialize the system from the sidebar first.")
    else:
        query_text = st.text_input(
            "Enter your question:",
            placeholder="e.g., What are the customer due diligence requirements?",
            key="query_input"
        )
        
        col1, col2 = st.columns([1, 4])
        with col1:
            top_k = st.slider("Top K Results", 1, 10, 5)
        with col2:
            use_llm = st.checkbox("Use Gemini for Answer Generation", value=True)
        
        if st.button("Run Analysis", type="primary") and query_text:
            with st.spinner("Reviewing corpus and drafting response..."):
                result = query(query_text, top_k=top_k, use_llm=use_llm)
            
            st.markdown("### AI Response")
            if result['answer'].lower().startswith("error"):
                st.error(result['answer'])
            else:
                st.success(result['answer'])
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Confidence", f"{result['confidence']:.3f}")
            with col2:
                st.metric("Chunks Utilized", result.get('num_chunks_retrieved', len(result['citations'])))
            with col3:
                st.metric("Citations Returned", len(result['citations']))
            
            st.markdown("### Supporting Evidence")
            for citation in result['citations']:
                with st.expander(f"[{citation['citation_id']}] {citation['source']} (Similarity: {citation['similarity_score']:.3f})", expanded=False):
                    st.write(citation['text_preview'])

# Tab 2: Narrative Generator
with tab2:
    st.header("Generate Compliance Narrative")
    
    if not st.session_state.system_initialized:
        st.warning("Please initialize the system from the sidebar first.")
    else:
        st.markdown("Generate audit narratives for flagged transactions/compliance exceptions")
        
        col1, col2 = st.columns(2)
        with col1:
            transaction_amount = st.number_input(
                "Transaction Amount ($)",
                min_value=0.0,
                value=50000.0,
                step=1000.0
            )
            client_profile = st.text_area(
                "Client Profile",
                placeholder="e.g., New individual customer, high-risk jurisdiction, cash deposit",
                height=100
            )
        
        with col2:
            flagged_rule = st.text_input(
                "Flagged Rule",
                placeholder="e.g., Section 13(e) - Transactions above $50,000 require enhanced due diligence"
            )
            additional_context = st.text_area(
                "Additional Context (Optional)",
                placeholder="e.g., Customer declined to provide additional documentation.",
                height=100
            )
        
        if st.button("Generate Narrative", type="primary"):
            if not client_profile or not flagged_rule:
                st.error("Please fill in Client Profile and Flagged Rule")
            else:
                with st.spinner("Generating narrative..."):
                    narrative_result = generate_narrative(
                        transaction_amount=transaction_amount,
                        client_profile=client_profile,
                        flagged_rule=flagged_rule,
                        additional_context=additional_context if additional_context else None
                    )
                
                st.subheader("Generated Narrative")
                st.write(narrative_result['narrative'])
                
                # Download button
                st.download_button(
                    label="Download Narrative",
                    data=json.dumps(narrative_result, indent=2),
                    file_name=f"compliance_narrative_{narrative_result['generated_at'][:10]}.json",
                    mime="application/json"
                )

# Tab 3: Testing
with tab3:
    st.header("Test RAG System")
    
    if not st.session_state.system_initialized:
        st.warning("Please initialize the system from the sidebar first.")
    else:
        st.markdown("Run test suite to evaluate citation accuracy and hallucination rate")
        
        # Load test questions
        test_questions_file = st.file_uploader(
            "Upload Test Questions (JSON)",
            type=['json'],
            help="Upload a JSON file with test questions."
        )
        
        if test_questions_file:
            questions = json.load(test_questions_file)
        else:
            questions = load_test_questions()
            st.info(f"Using default test questions ({len(questions)} questions)")
        
        use_llm_test = st.checkbox("Use Gemini for Testing", value=False)
        
        if st.button("Run Test Suite", type="primary"):
            with st.spinner("Running tests..."):
                results = run_test_suite(questions, use_llm=use_llm_test)
            
            # Display metrics
            metrics = results['metrics']
            
            st.subheader("Test Results Summary")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Questions", metrics['total_questions'])
            with col2:
                st.metric("Citation Rate", f"{metrics['citation_rate']:.1%}")
            with col3:
                st.metric("Avg Citation Quality", f"{metrics['avg_citation_quality']:.2f}")
            with col4:
                st.metric("Hallucination Mitigation", f"{metrics['hallucination_mitigation_rate']:.1%}")
            
            # Detailed results
            st.subheader("Detailed Results")
            for i, result in enumerate(results['results'], 1):
                with st.expander(f"[{i}] {result['question']}"):
                    st.write(f"**Answer:** {result['answer'][:500]}...")
                    st.write(f"**Citations:** {result['num_citations']}")
                    st.write(f"**Confidence:** {result['confidence']:.3f}")
                    st.write(f"**Keyword Coverage:** {result['keyword_coverage']:.1%}")
                    st.write(f"**Hallucination Risk:** {result['hallucination_indicators']['hallucination_risk']}")
            
            # Download results
            st.download_button(
                label="Download Test Results",
                data=json.dumps(results, indent=2),
                file_name="test_results.json",
                mime="application/json"
            )

# Tab 4: Status
with tab4:
    st.header("System Status")
    
    # Check components
    components = {
        "PDF Extraction": os.path.exists('pdf_extractor.py'),
        "Text Chunking": os.path.exists('text_chunker.py'),
        "RAG Pipeline": os.path.exists('rag_pipeline.py'),
        "Vector Store": os.path.exists('vector_store/faiss_index.bin'),
        "Chunk Metadata": os.path.exists('vector_store/chunk_metadata.pkl'),
        "Embeddings Info": os.path.exists('vector_store/embeddings_info.json'),
        "Query System": os.path.exists('rag_query_system.py'),
        "Narrative Generator": os.path.exists('narrative_generator.py'),
        "Testing Framework": os.path.exists('test_rag_system.py'),
    }
    
    st.subheader("Component Status")
    for component, exists in components.items():
        if exists:
            st.success(f"✓ {component}")
        else:
            st.error(f"✗ {component}")
    
    # Vector store info
    if os.path.exists('vector_store/embeddings_info.json'):
        st.subheader("Vector Store Information")
        with open('vector_store/embeddings_info.json', 'r') as f:
            info = json.load(f)
        st.json(info)
    
    # System requirements
    st.subheader("System Requirements")
    st.markdown("""
    **Required Packages:**
    - `pymupdf` - PDF extraction
    - `sentence-transformers` - Embeddings
    - `faiss-cpu` or `faiss-gpu` - Vector store
    - `langchain` - RAG pipeline
    - `langchain-google-genai` - Gemini integration
    - `streamlit` - UI
    - `numpy` - Numerical operations
    
    **Installation:**
    ```bash
    pip install -r requirements.txt
    ```
    """)
