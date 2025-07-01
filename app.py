import streamlit as st
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import pickle
import os
from langchain_community.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Set page config
st.set_page_config(
    page_title="ChaosRAG - Landmine Identification System",
    page_icon="⚠️",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #ff6b6b, #feca57);
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin-bottom: 2rem;
    }
    .warning-box {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .result-box {
        background-color: #000000;
        border: 1px solid #c3e6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .danger-box {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        padding: 1rem;
        border-radius: 5px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Title and header
st.markdown("""
<div class="main-header">
    <h1>⚠️ ChaosRAG - Landmine Identification System</h1>
    <p>AI-Powered Image Classification & Knowledge Retrieval for EOD Operations</p>
</div>
""", unsafe_allow_html=True)

# Safety warning
st.markdown("""
<div class="danger-box">
    <h3>🚨 CRITICAL SAFETY NOTICE</h3>
    <p><strong>This system is for educational and training purposes only. Always follow proper EOD protocols and consult with certified EOD specialists before handling any suspected explosive devices.</strong></p>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_classification_model():
    """Load the image classification model"""
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        classes = ['ADM-8', 'DKRP-4_MinefieldClearance', 'Dog_Landmine', 'IGS-50', 'MOB_Landmine', 
                  'MON-100', 'MON-200', 'MON-50', 'MON-90', 'MZM-2', 'MZU-S', 'OMZ-4', 
                  'OZM-160', 'OZM_Landmine', 'PFM-1', 'PMD-6&7', 'PMK-1_Portable_Mining_Kit', 
                  'PMN-2', 'PMN-3', 'PMN-4', 'PMN_Landmine', 'PMP_Landmine', 'POB-Pilka', 
                  'POM-1', 'POM-2', 'POM-3', 'POMZ-2', 'POMZ-2M', 'PTKM-1R', 'PTM-1S', 
                  'PTM-25', 'PTM-3', 'PTM-4', 'PVM_Landmine', 'SM-320_Signnal_Flare', 'TM-35', 
                  'TM-35M', 'TM-38', 'TM-39', 'TM-41', 'TM-46', 'TM-56', 'TM-57', 'TM-62B', 
                  'TM-62D', 'TM-62M', 'TM-62P', 'TM-62P2', 'TM-62P3', 'TM-62T', 'TM-71', 
                  'TM-83', 'TM-89', 'TMB-1', 'TMD-B', 'TMK-2', 'Temp-30', 'UDSh_Landmine', 
                  'UR-83P_Minefield_Clearance', 'Yam_Landmine', 'ZRP-2_Minefield_Clearance', 
                  'ozm-3', 'ozm-72', 'pmk-40', 'pmm-3', 'pmm-5', 'pmz-40']
        
        # Define the model architecture
        model = models.resnet18(pretrained=True)
        
        # Modify the architecture as in the original notebook
        model.layer4[1].conv2 = nn.Sequential(
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.4)
        )
        model.layer4[1].bn2 = nn.Identity()
        model.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        model.fc = nn.Linear(512, len(classes))
        
        # Unfreeze all layers
        for param in model.parameters():
            param.requires_grad = True
        
        # Load the saved state dictionary
        checkpoint_path = "best_resnet18.pth"
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(device)
            model.eval()
            
            return model, classes, device
        else:
            st.error(f"Model file {checkpoint_path} not found!")
            return None, None, None
            
    except Exception as e:
        st.error(f"Error loading classification model: {str(e)}")
        return None, None, None

@st.cache_resource
def load_rag_system():
    """Load the RAG system components"""
    try:
        # Load embeddings and retriever
        embeddings = OllamaEmbeddings(model="nomic-embed-text")
        
        # Try to load existing retriever
        retriever_path = "retriever_Landmines.pkl"
        vector_store_path = "Landmines_faiss_index_ollama"
        
        if os.path.exists(retriever_path):
            with open(retriever_path, 'rb') as f:
                retriever = pickle.load(f)
        elif os.path.exists(vector_store_path):
            persisted_vectorstore = FAISS.load_local(vector_store_path, embeddings, allow_dangerous_deserialization=True)
            retriever = persisted_vectorstore.as_retriever()
        else:
            st.error("No retriever or vector store found!")
            return None, None
        
        # Initialize LLM
        llm = Ollama(model="llama3.1")
        
        # Define custom prompt template
        template = """You are a specialized explosive ordnance disposal (EOD) expert and landmine identification specialist. 
Your role is to assist defense personnel in identifying landmines based on field descriptions and available intelligence or 
provide all known information based on the name/type of the landmine.

Context: {context}

Field Description/Question: {question}

Expert Analysis:"""

        prompt = PromptTemplate(
            template=template, 
            input_variables=["context", "question"]
        )

        # Create RetrievalQA with custom prompt
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa, retriever
        
    except Exception as e:
        st.error(f"Error loading RAG system: {str(e)}")
        return None, None

def classify_image(image, model, classes, device):
    """Classify uploaded image and return top 5 predictions"""
    try:
        # Define image transformation
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Apply transformations
        image_tensor = transform(image).unsqueeze(0)
        image_tensor = image_tensor.to(device)
        
        # Perform inference
        with torch.no_grad():
            outputs = model(image_tensor)
        
        # Get top 5 predictions
        probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
        _, top5_idx = torch.topk(outputs, 5)
        top5_classes = [classes[idx.item()] for idx in top5_idx[0]]
        top5_probs = [probabilities[idx.item()].item() for idx in top5_idx[0]]
        
        return top5_classes, top5_probs
        
    except Exception as e:
        st.error(f"Error during classification: {str(e)}")
        return None, None

def get_landmine_info(landmine_name, qa_system):
    """Get detailed information about a landmine using RAG"""
    try:
        result = qa_system(landmine_name)
        return result["result"], result.get("source_documents", [])
    except Exception as e:
        st.error(f"Error retrieving landmine information: {str(e)}")
        return None, None

# Load models
model, classes, device = load_classification_model()
qa_system, retriever = load_rag_system()

# Main application layout
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📸 Image Classification")
    st.markdown("Upload an image of a suspected landmine for AI-powered identification.")
    
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png', 'bmp'],
        help="Upload a clear image of the suspected landmine"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        if model is not None and classes is not None:
            with st.spinner("Analyzing image..."):
                top5_classes, top5_probs = classify_image(image, model, classes, device)
            
            if top5_classes:
                st.markdown("### 🎯 Top 5 Predictions:")
                
                # Create a selectbox for the predictions
                prediction_options = []
                for i, (class_name, prob) in enumerate(zip(top5_classes, top5_probs)):
                    prediction_options.append(f"{class_name} ({prob:.2%} confidence)")
                
                selected_prediction = st.selectbox(
                    "Select a classification result for detailed information:",
                    prediction_options,
                    help="Choose from the top 5 predictions to get detailed information"
                )
                
                # Extract the class name from the selected option
                selected_class = selected_prediction.split(" (")[0]
                
                # Display classification results
                st.markdown("#### Classification Results:")
                for i, (class_name, prob) in enumerate(zip(top5_classes, top5_probs)):
                    confidence_color = "🟢" if prob > 0.7 else "🟡" if prob > 0.4 else "🔴"
                    # st.markdown(f"{i+1}. **{class_name}** - {confidence_color} {prob:.2%}")
                    st.markdown(f"{i+1}. **{class_name}**")

with col2:
    st.header("🧠 Knowledge Retrieval")
    st.markdown("Get detailed information about landmines using our AI knowledge base.")
    
    # Manual query option
    manual_query = st.text_input(
        "Or enter a landmine name/description manually:",
        placeholder="e.g., PMK-40, TM-62M, etc.",
        help="Enter the name or description of a landmine"
    )
    
    # Determine which query to use
    query_to_use = None
    if uploaded_file is not None and 'selected_class' in locals():
        query_to_use = selected_class
        st.info(f"Using classification result: **{selected_class}**")
    elif manual_query:
        query_to_use = manual_query
        st.info(f"Using manual query: **{manual_query}**")
    
    # Get landmine information
    if query_to_use and qa_system:
        with st.spinner("Retrieving information..."):
            info, sources = get_landmine_info(query_to_use, qa_system)
        
        if info:
            st.markdown("### 📋 Expert Analysis:")
            st.markdown(f"""
            <div class="result-box">
                {info}
            </div>
            """, unsafe_allow_html=True)
            
            # Show source documents
            if sources:
                with st.expander("📚 Source Documents"):
                    for i, doc in enumerate(sources):
                        st.markdown(f"**Source {i+1}:**")
                        st.markdown(f"Page: {doc.metadata.get('page', 'Unknown')}")
                        st.markdown(f"Content: {doc.page_content[:300]}...")
                        st.markdown("---")

# Sidebar with additional information
with st.sidebar:
    st.header("ℹ️ System Information")
    
    st.markdown("### Model Status:")
    if model is not None:
        st.success("✅ Classification Model Loaded")
    else:
        st.error("❌ Classification Model Failed to Load")
    
    if qa_system is not None:
        st.success("✅ RAG System Loaded")
    else:
        st.error("❌ RAG System Failed to Load")
    
    device_info = "CUDA" if torch.cuda.is_available() else "CPU"
    st.info(f"🖥️ Running on: {device_info}")
    
    st.markdown("### 📖 Usage Instructions:")
    st.markdown("""
    1. **Upload Image**: Upload a clear image of the suspected landmine
    2. **Review Predictions**: Check the top 5 classification results
    3. **Select Prediction**: Choose a prediction from the dropdown
    4. **Get Information**: Detailed information will appear automatically
    5. **Manual Query**: You can also search manually by name
    """)
    
    st.markdown("### ⚠️ Safety Reminders:")
    st.markdown("""
    - Never approach suspected explosives
    - Always contact EOD specialists
    - This system is for training purposes only
    - Follow all safety protocols
    """)
    
    st.markdown("### 🔧 Supported Landmine Types:")
    if classes:
        st.markdown(f"The system can identify {len(classes)} different landmine types.")
        with st.expander("View all supported types"):
            for class_name in sorted(classes):
                st.markdown(f"• {class_name}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <p>ChaosRAG Landmine Identification System | For Educational and Training Purposes Only</p>
    <p>⚠️ Always consult with certified EOD specialists for real-world operations</p>
</div>
""", unsafe_allow_html=True) 