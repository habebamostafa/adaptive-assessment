import streamlit as st
import json
import time
import random
from typing import List, Dict

class MCQGenerator:
    def __init__(self):
        """Initialize the MCQ generator"""
        self.available_tracks = {
            "web": "Web Development (HTML, CSS, JavaScript, React, Vue, Angular)",
            "ai": "Artificial Intelligence (Machine Learning, Deep Learning, NLP)",
            "cyber": "Cybersecurity (Network Security, Encryption, Ethical Hacking)",
            "data": "Data Science (Data Analysis, Visualization, Statistics, Python)",
            "mobile": "Mobile Development (Android, iOS, Flutter, React Native)",
            "devops": "DevOps (Docker, Kubernetes, CI/CD, Cloud Computing)",
            "backend": "Backend Development (APIs, Databases, Microservices)",
            "frontend": "Frontend Development (UI/UX, Responsive Design)"
        }
        
    def get_comprehensive_demo_questions(self):
        """Comprehensive demo question database"""
        return {
            "web": {
                "easy": [
                    {
                        "question": "What does CSS stand for?",
                        "options": ["Cascading Style Sheets", "Computer Style Sheets", "Creative Style Sheets", "Colorful Style Sheets"],
                        "correct_answer": "Cascading Style Sheets",
                        "explanation": "CSS stands for Cascading Style Sheets, used to describe the presentation of HTML documents including layout, colors, and fonts."
                    },
                    {
                        "question": "Which HTML tag is used for the largest heading?",
                        "options": ["<h1>", "<h6>", "<header>", "<heading>"],
                        "correct_answer": "<h1>",
                        "explanation": "The <h1> tag represents the largest/most important heading in HTML, with headings decreasing in size from h1 to h6."
                    },
                    {
                        "question": "What does JavaScript primarily add to web pages?",
                        "options": ["Interactivity and dynamic behavior", "Styling and layout", "Structure and content", "Database connectivity"],
                        "correct_answer": "Interactivity and dynamic behavior",
                        "explanation": "JavaScript is a programming language that adds interactive elements and dynamic functionality to web pages."
                    },
                    {
                        "question": "Which CSS property is used to change the text color?",
                        "options": ["color", "text-color", "font-color", "text-style"],
                        "correct_answer": "color",
                        "explanation": "The 'color' property in CSS is used to set the color of text."
                    },
                    {
                        "question": "What is the purpose of the <div> tag in HTML?",
                        "options": ["To define a division or section", "To create a dynamic element", "To add styling", "To include JavaScript"],
                        "correct_answer": "To define a division or section",
                        "explanation": "The <div> tag is used as a container for HTML elements and is often used to group elements for styling purposes."
                    }
                ],
                "medium": [
                    {
                        "question": "What is the main purpose of React's useState hook?",
                        "options": ["Manage component state in functional components", "Handle HTTP requests", "Style components", "Route between pages"],
                        "correct_answer": "Manage component state in functional components",
                        "explanation": "useState is a React hook that allows functional components to have and update state, eliminating the need for class components in many cases."
                    },
                    {
                        "question": "What is the difference between '==' and '===' in JavaScript?",
                        "options": ["=== checks type and value, == only checks value", "== is faster than ===", "=== is deprecated", "No difference"],
                        "correct_answer": "=== checks type and value, == only checks value",
                        "explanation": "=== performs strict equality checking both type and value, while == performs loose equality with type coercion."
                    },
                    {
                        "question": "What is the purpose of CSS Grid?",
                        "options": ["Create two-dimensional layouts", "Add animations", "Handle responsive images", "Manage fonts"],
                        "correct_answer": "Create two-dimensional layouts",
                        "explanation": "CSS Grid is a layout system that allows you to create complex two-dimensional layouts with rows and columns."
                    },
                    {
                        "question": "What is a closure in JavaScript?",
                        "options": ["A function with access to its outer function's scope", "A way to close browser tabs", "A CSS animation technique", "A React component lifecycle method"],
                        "correct_answer": "A function with access to its outer function's scope",
                        "explanation": "A closure is a function that has access to its own scope, the outer function's scope, and the global scope."
                    },
                    {
                        "question": "What is the virtual DOM in React?",
                        "options": ["A lightweight copy of the real DOM", "A security feature", "A database for storing components", "A browser API"],
                        "correct_answer": "A lightweight copy of the real DOM",
                        "explanation": "The virtual DOM is a programming concept where a virtual representation of the UI is kept in memory and synced with the real DOM."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the event loop in JavaScript?",
                        "options": ["Mechanism for handling asynchronous operations", "A type of HTML element", "A CSS animation property", "A React lifecycle method"],
                        "correct_answer": "Mechanism for handling asynchronous operations",
                        "explanation": "The event loop is JavaScript's concurrency model that handles asynchronous callbacks and ensures non-blocking execution."
                    },
                    {
                        "question": "What is webpack's code splitting feature?",
                        "options": ["Divide code into smaller bundles for better performance", "Separate CSS from JavaScript", "Split development and production builds", "Divide frontend from backend"],
                        "correct_answer": "Divide code into smaller bundles for better performance",
                        "explanation": "Code splitting allows webpack to break your code into smaller chunks that can be loaded on demand, improving initial load times."
                    },
                    {
                        "question": "What is the purpose of the Shadow DOM?",
                        "options": ["Encapsulate DOM and CSS for web components", "Create shadows in UI design", "Improve performance of DOM operations", "Add security to web applications"],
                        "correct_answer": "Encapsulate DOM and CSS for web components",
                        "explanation": "Shadow DOM provides encapsulation for the DOM and CSS of web components, preventing styles from leaking out and external styles from affecting the component."
                    },
                    {
                        "question": "What is the difference between HTTP/1.1 and HTTP/2?",
                        "options": ["HTTP/2 allows multiplexing while HTTP/1.1 does not", "HTTP/2 is slower than HTTP/1.1", "HTTP/2 doesn't support HTTPS", "HTTP/2 is only for mobile applications"],
                        "correct_answer": "HTTP/2 allows multiplexing while HTTP/1.1 does not",
                        "explanation": "HTTP/2 introduces multiplexing, which allows multiple requests and responses to be sent simultaneously over a single connection, improving performance."
                    },
                    {
                        "question": "What is Tree Shaking in JavaScript?",
                        "options": ["Removing unused code from bundles", "A data structure algorithm", "A DOM manipulation technique", "A React optimization method"],
                        "correct_answer": "Removing unused code from bundles",
                        "explanation": "Tree shaking is a term commonly used in the JavaScript context for dead-code elimination, which helps reduce bundle size by removing unused code."
                    }
                ]
            },
            "ai": {
                "easy": [
                    {
                        "question": "What is artificial intelligence?",
                        "options": ["Computer systems that can perform tasks requiring human intelligence", "Only robots that look like humans", "Software for data storage", "Internet search engines"],
                        "correct_answer": "Computer systems that can perform tasks requiring human intelligence",
                        "explanation": "AI refers to computer systems capable of performing tasks that typically require human intelligence, such as learning, reasoning, and problem-solving."
                    },
                    {
                        "question": "What type of learning uses labeled training data?",
                        "options": ["Supervised learning", "Unsupervised learning", "Reinforcement learning", "Deep learning"],
                        "correct_answer": "Supervised learning",
                        "explanation": "Supervised learning uses labeled datasets where the correct output is known, allowing the model to learn the relationship between inputs and outputs."
                    },
                    {
                        "question": "What is a neural network?",
                        "options": ["A computing system inspired by the human brain", "A type of computer hardware", "A network of computers", "A database management system"],
                        "correct_answer": "A computing system inspired by the human brain",
                        "explanation": "Neural networks are computing systems inspired by the biological neural networks that constitute animal brains, consisting of interconnected nodes that process information."
                    },
                    {
                        "question": "What is the main goal of machine learning?",
                        "options": ["Enable computers to learn from data without explicit programming", "Create faster algorithms", "Build better user interfaces", "Improve computer hardware"],
                        "correct_answer": "Enable computers to learn from data without explicit programming",
                        "explanation": "Machine learning focuses on developing algorithms that allow computers to learn from and make predictions or decisions based on data."
                    },
                    {
                        "question": "What is natural language processing (NLP)?",
                        "options": ["AI field focused on interaction between computers and human language", "A programming language", "A database query language", "A network protocol"],
                        "correct_answer": "AI field focused on interaction between computers and human language",
                        "explanation": "NLP is a subfield of AI that focuses on enabling computers to understand, interpret, and generate human language."
                    }
                ],
                "medium": [
                    {
                        "question": "What is overfitting in machine learning?",
                        "options": ["Model performs well on training data but poorly on new data", "Model trains too slowly", "Model uses too much memory", "Model cannot learn any patterns"],
                        "correct_answer": "Model performs well on training data but poorly on new data",
                        "explanation": "Overfitting occurs when a model learns the training data too specifically, including noise, making it perform poorly on unseen data."
                    },
                    {
                        "question": "What is the purpose of activation functions in neural networks?",
                        "options": ["Introduce non-linearity to enable complex learning", "Store training data", "Connect to databases", "Display results to users"],
                        "correct_answer": "Introduce non-linearity to enable complex learning",
                        "explanation": "Activation functions add non-linearity to neural networks, allowing them to learn complex patterns and relationships in data."
                    },
                    {
                        "question": "What is gradient descent?",
                        "options": ["An optimization algorithm to minimize loss functions", "A data visualization technique", "A type of neural network architecture", "A database indexing method"],
                        "correct_answer": "An optimization algorithm to minimize loss functions",
                        "explanation": "Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent as defined by the negative of the gradient."
                    },
                    {
                        "question": "What is a convolutional neural network (CNN) primarily used for?",
                        "options": ["Image recognition and processing", "Text generation", "Audio synthesis", "Database management"],
                        "correct_answer": "Image recognition and processing",
                        "explanation": "CNNs are a class of deep neural networks most commonly applied to analyzing visual imagery, leveraging spatial relationships in data."
                    },
                    {
                        "question": "What is the difference between classification and regression?",
                        "options": ["Classification predicts categories, regression predicts continuous values", "Classification is faster than regression", "Regression is for images, classification is for text", "No significant difference"],
                        "correct_answer": "Classification predicts categories, regression predicts continuous values",
                        "explanation": "Classification models predict discrete class labels, while regression models predict continuous quantities."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the vanishing gradient problem?",
                        "options": ["Gradients become very small in early layers during backpropagation", "Model outputs become zero", "Training data disappears", "Network connections break"],
                        "correct_answer": "Gradients become very small in early layers during backpropagation",
                        "explanation": "The vanishing gradient problem occurs when gradients become exponentially smaller as they propagate backward through deep networks, making early layers difficult to train."
                    },
                    {
                        "question": "What is transfer learning?",
                        "options": ["Using knowledge from one task to improve learning in another", "Moving data between databases", "Transferring files between servers", "Changing programming languages"],
                        "correct_answer": "Using knowledge from one task to improve learning in another",
                        "explanation": "Transfer learning is a machine learning method where a model developed for a task is reused as the starting point for a model on a second task."
                    },
                    {
                        "question": "What is the attention mechanism in neural networks?",
                        "options": ["Focusing on relevant parts of input when producing output", "A user interface feature", "A data storage technique", "A network security protocol"],
                        "correct_answer": "Focusing on relevant parts of input when producing output",
                        "explanation": "The attention mechanism allows models to focus on different parts of the input sequence when producing each part of the output sequence, improving performance on tasks like translation."
                    },
                    {
                        "question": "What is a generative adversarial network (GAN)?",
                        "options": ["A system of two neural networks competing against each other", "A security network", "A database architecture", "A type of computer virus"],
                        "correct_answer": "A system of two neural networks competing against each other",
                        "explanation": "GANs consist of two neural networks, a generator and a discriminator, that compete against each other to generate new, synthetic data that resembles real data."
                    },
                    {
                        "question": "What is the curse of dimensionality?",
                        "options": ["Problems that arise when working with high-dimensional data", "A programming bug", "A hardware limitation", "A network congestion issue"],
                        "correct_answer": "Problems that arise when working with high-dimensional data",
                        "explanation": "The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces that do not occur in low-dimensional settings."
                    }
                ]
            },
            "cyber": {
                "easy": [
                    {
                        "question": "What is malware?",
                        "options": ["Malicious software designed to harm computers", "A type of firewall", "Network monitoring tool", "Data backup system"],
                        "correct_answer": "Malicious software designed to harm computers",
                        "explanation": "Malware (malicious software) includes viruses, trojans, ransomware, and other programs designed to damage, disrupt, or gain unauthorized access to computer systems."
                    },
                    {
                        "question": "What is the primary purpose of encryption?",
                        "options": ["Protect data confidentiality", "Speed up data transfer", "Reduce file sizes", "Organize files"],
                        "correct_answer": "Protect data confidentiality",
                        "explanation": "Encryption converts readable data into an encoded format to protect confidentiality and prevent unauthorized access to sensitive information."
                    },
                    {
                        "question": "What is a firewall?",
                        "options": ["A network security system that monitors traffic", "A physical barrier for servers", "A type of virus", "A data storage device"],
                        "correct_answer": "A network security system that monitors traffic",
                        "explanation": "A firewall is a network security device that monitors and filters incoming and outgoing network traffic based on an organization's previously established security policies."
                    },
                    {
                        "question": "What is phishing?",
                        "options": ["Fraudulent attempt to obtain sensitive information", "A fishing game", "A type of encryption", "A network protocol"],
                        "correct_answer": "Fraudulent attempt to obtain sensitive information",
                        "explanation": "Phishing is a cyber attack that uses disguised email as a weapon, tricking the email recipient into believing the message is something they want or need."
                    },
                    {
                        "question": "What is two-factor authentication (2FA)?",
                        "options": ["Using two different verification methods for security", "Having two passwords", "Using two different browsers", "Logging in twice"],
                        "correct_answer": "Using two different verification methods for security",
                        "explanation": "2FA adds an extra layer of security by requiring two different authentication factors, such as something you know (password) and something you have (phone)."
                    }
                ],
                "medium": [
                    {
                        "question": "What is a Man-in-the-Middle (MITM) attack?",
                        "options": ["Intercepting communication between two parties", "Physical theft of computers", "Overloading servers with requests", "Installing malware via email"],
                        "correct_answer": "Intercepting communication between two parties",
                        "explanation": "A MITM attack occurs when an attacker secretly intercepts and potentially alters communication between two parties who believe they are communicating directly."
                    },
                    {
                        "question": "What is a DDoS attack?",
                        "options": ["Overwhelming a system with traffic to make it unavailable", "Stealing data from a database", "Encrypting files for ransom", "Creating fake websites"],
                        "correct_answer": "Overwhelming a system with traffic to make it unavailable",
                        "explanation": "A Distributed Denial of Service (DDoS) attack attempts to disrupt normal traffic of a targeted server, service or network by overwhelming it with a flood of Internet traffic."
                    },
                    {
                        "question": "What is SQL injection?",
                        "options": ["Injecting malicious code into SQL statements", "A database backup method", "A type of encryption", "A network routing technique"],
                        "correct_answer": "Injecting malicious code into SQL statements",
                        "explanation": "SQL injection is a code injection technique that might destroy your database by inserting malicious SQL statements into an entry field for execution."
                    },
                    {
                        "question": "What is the difference between symmetric and asymmetric encryption?",
                        "options": ["Symmetric uses one key, asymmetric uses two keys", "Symmetric is faster, asymmetric is slower", "Symmetric is for email, asymmetric is for files", "No significant difference"],
                        "correct_answer": "Symmetric uses one key, asymmetric uses two keys",
                        "explanation": "Symmetric encryption uses the same key for encryption and decryption, while asymmetric encryption uses a public key for encryption and a private key for decryption."
                    },
                    {
                        "question": "What is a zero-day vulnerability?",
                        "options": ["A security flaw unknown to the vendor", "A vulnerability that appears at midnight", "A bug that fixes itself in one day", "A minor security issue"],
                        "correct_answer": "A security flaw unknown to the vendor",
                        "explanation": "A zero-day vulnerability is a software security flaw that is unknown to those who should be interested in mitigating the vulnerability, including the vendor."
                    }
                ],
                "hard": [
                    {
                        "question": "What is a zero-day exploit?",
                        "options": ["Attack using previously unknown vulnerabilities", "Attack that happens at midnight", "Attack that takes zero time", "Attack that costs no money"],
                        "correct_answer": "Attack using previously unknown vulnerabilities",
                        "explanation": "A zero-day exploit takes advantage of a security vulnerability that is unknown to security vendors and has no available patch or defense."
                    },
                    {
                        "question": "What is penetration testing?",
                        "options": ["Authorized simulated cyber attack on a system", "Testing pen durability", "A type of encryption", "A network monitoring tool"],
                        "correct_answer": "Authorized simulated cyber attack on a system",
                        "explanation": "Penetration testing, also called pen testing, is a simulated cyber attack against your computer system to check for exploitable vulnerabilities."
                    },
                    {
                        "question": "What is the principle of least privilege?",
                        "options": ["Users should have only necessary access permissions", "All users should have admin rights", "Systems should have maximum security", "Networks should be completely open"],
                        "correct_answer": "Users should have only necessary access permissions",
                        "explanation": "The principle of least privilege recommends that users and systems should have the minimum levels of access necessary to perform their functions."
                    },
                    {
                        "question": "What is social engineering in cybersecurity?",
                        "options": ["Manipulating people to divulge confidential information", "Engineering social networks", "Building secure systems", "Designing user interfaces"],
                        "correct_answer": "Manipulating people to divulge confidential information",
                        "explanation": "Social engineering is the psychological manipulation of people into performing actions or divulging confidential information, rather than by breaking in or using technical cracking techniques."
                    },
                    {
                        "question": "What is a honeypot in cybersecurity?",
                        "options": ["A trap set to detect or deflect unauthorized access", "A sweet security solution", "A type of encryption", "A user authentication method"],
                        "correct_answer": "A trap set to detect or deflect unauthorized access",
                        "explanation": "A honeypot is a security mechanism that creates a virtual trap to lure attackers, allowing security professionals to study their methods and gather intelligence."
                    }
                ]
            },
            "data": {
                "easy": [
                    {
                        "question": "What is data science?",
                        "options": ["Extracting insights and knowledge from data", "Just creating charts and graphs", "Only working with big data", "Programming databases"],
                        "correct_answer": "Extracting insights and knowledge from data",
                        "explanation": "Data science combines statistics, programming, and domain expertise to extract meaningful insights and knowledge from structured and unstructured data."
                    },
                    {
                        "question": "What is the purpose of data visualization?",
                        "options": ["Communicate information clearly through graphical means", "Store data efficiently", "Encrypt data for security", "Delete unnecessary data"],
                        "correct_answer": "Communicate information clearly through graphical means",
                        "explanation": "Data visualization is the graphical representation of information and data to communicate relationships and insights effectively."
                    },
                    {
                        "question": "What is a database?",
                        "options": ["Organized collection of structured information", "A type of computer hardware", "A programming language", "A network protocol"],
                        "correct_answer": "Organized collection of structured information",
                        "explanation": "A database is an organized collection of structured information, or data, typically stored electronically in a computer system."
                    },
                    {
                        "question": "What is SQL used for?",
                        "options": ["Managing and querying relational databases", "Creating web pages", "Designing user interfaces", "Writing operating systems"],
                        "correct_answer": "Managing and querying relational databases",
                        "explanation": "SQL (Structured Query Language) is a programming language designed for managing data held in a relational database management system."
                    },
                    {
                        "question": "What is the difference between structured and unstructured data?",
                        "options": ["Structured data is organized, unstructured is not", "Structured data is larger", "Unstructured data is easier to analyze", "No significant difference"],
                        "correct_answer": "Structured data is organized, unstructured is not",
                        "explanation": "Structured data is highly organized and easily analyzed, while unstructured data has no pre-defined format or organization, making it more difficult to analyze."
                    }
                ],
                "medium": [
                    {
                        "question": "What is the purpose of data normalization?",
                        "options": ["Scale features to similar ranges", "Remove duplicate data", "Convert text to numbers", "Compress data files"],
                        "correct_answer": "Scale features to similar ranges",
                        "explanation": "Data normalization scales numerical features to similar ranges, preventing features with larger scales from dominating the analysis or model training."
                    },
                    {
                        "question": "What is a correlation coefficient?",
                        "options": ["Measure of the strength of relationship between variables", "A database index", "A type of algorithm", "A data storage format"],
                        "correct_answer": "Measure of the strength of relationship between variables",
                        "explanation": "A correlation coefficient is a numerical measure that describes the strength and direction of the relationship between two variables."
                    },
                    {
                        "question": "What is the difference between supervised and unsupervised learning?",
                        "options": ["Supervised uses labeled data, unsupervised uses unlabeled data", "Supervised is faster", "Unsupervised is more accurate", "No significant difference"],
                        "correct_answer": "Supervised uses labeled data, unsupervised uses unlabeled data",
                        "explanation": "Supervised learning uses labeled training data, while unsupervised learning finds patterns in unlabeled data without pre-existing labels."
                    },
                    {
                        "question": "What is overfitting in data science?",
                        "options": ["Model learns training data too well but performs poorly on new data", "Model is too simple", "Model uses too much memory", "Model cannot learn patterns"],
                        "correct_answer": "Model learns training data too well but performs poorly on new data",
                        "explanation": "Overfitting occurs when a model learns the detail and noise in the training data to the extent that it negatively impacts the performance on new data."
                    },
                    {
                        "question": "What is cross-validation?",
                        "options": ["Technique to evaluate model performance on unseen data", "A data cleaning method", "A database optimization technique", "A data visualization approach"],
                        "correct_answer": "Technique to evaluate model performance on unseen data",
                        "explanation": "Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample, providing insight into how the model will generalize to an independent dataset."
                    }
                ],
                "hard": [
                    {
                        "question": "What is the bias-variance tradeoff?",
                        "options": ["Balance between model simplicity and complexity", "Choice between speed and accuracy", "Tradeoff between training and testing", "Balance between data size and model size"],
                        "correct_answer": "Balance between model simplicity and complexity",
                        "explanation": "The bias-variance tradeoff describes the relationship between a model's ability to minimize bias (underfitting) and variance (overfitting) to achieve optimal predictive performance."
                    },
                    {
                        "question": "What is feature engineering?",
                        "options": ["Process of creating new input features from existing data", "Engineering software features", "Designing database schemas", "Building data pipelines"],
                        "correct_answer": "Process of creating new input features from existing data",
                        "explanation": "Feature engineering is the process of using domain knowledge to create features that make machine learning algorithms work better by transforming raw data into features that better represent the underlying problem."
                    },
                    {
                        "question": "What is ensemble learning?",
                        "options": ["Combining multiple models to improve performance", "Learning in groups", "A type of neural network", "A database clustering method"],
                        "correct_answer": "Combining multiple models to improve performance",
                        "explanation": "Ensemble learning combines multiple machine learning models to create a more accurate and robust predictive model than any of the individual models."
                    },
                    {
                        "question": "What is the curse of dimensionality?",
                        "options": ["Problems that arise when working with high-dimensional data", "A programming bug", "A hardware limitation", "A network congestion issue"],
                        "correct_answer": "Problems that arise when working with high-dimensional data",
                        "explanation": "The curse of dimensionality refers to various phenomena that arise when analyzing and organizing data in high-dimensional spaces that do not occur in low-dimensional settings."
                    },
                    {
                        "question": "What is transfer learning?",
                        "options": ["Using knowledge from one task to improve learning in another", "Moving data between databases", "Transferring files between servers", "Changing programming languages"],
                        "correct_answer": "Using knowledge from one task to improve learning in another",
                        "explanation": "Transfer learning is a machine learning technique where a model trained on one task is repurposed as the starting point for a model on a second related task."
                    }
                ]
            }
        }
    
    def get_demo_question(self, track: str, difficulty: str) -> Dict:
        """Get a demo question with rotation"""
        demo_db = self.get_comprehensive_demo_questions()
        
        # Initialize question indices
        if not hasattr(self, '_question_indices'):
            self._question_indices = {}
        
        key = f"{track}_{difficulty}"
        
        if track in demo_db and difficulty in demo_db[track]:
            questions = demo_db[track][difficulty]
            if questions:
                # Rotate through questions
                if key not in self._question_indices:
                    self._question_indices[key] = 0
                
                question_data = questions[self._question_indices[key] % len(questions)]
                self._question_indices[key] += 1
                
                return {
                    'text': question_data['question'],
                    'options': question_data['options'],
                    'correct_answer': question_data['correct_answer'],
                    'explanation': question_data['explanation'],
                    'track': track,
                    'difficulty': difficulty,
                    'generated_by': 'Question Database'
                }
        
        # Fallback generic question
        return {
            'text': f"What is a fundamental concept in {self.available_tracks.get(track, track)}?",
            'options': [
                "The core principle of this technology",
                "An unrelated software tool",
                "A hardware component only",
                "A deprecated technique"
            ],
            'correct_answer': "The core principle of this technology",
            'explanation': f"This question tests fundamental understanding of {self.available_tracks.get(track, track)} concepts at {difficulty} level.",
            'track': track,
            'difficulty': difficulty,
            'generated_by': 'Generic Fallback'
        }
    
    def generate_questions(self, track: str, difficulty: str, count: int) -> List[Dict]:
        """Generate multiple questions"""
        questions = []
        
        for i in range(count):
            question = self.get_demo_question(track, difficulty)
            questions.append(question)
        
        return questions

def main():
    st.set_page_config(
        page_title="MCQ Generator - No API Needed",
        page_icon="🎯",
        layout="wide"
    )
    
    st.title("🎯 MCQ Generator - No API Needed")
    st.markdown("Generate technical interview questions using our comprehensive question database")
    
    # Initialize generator
    if 'generator' not in st.session_state:
        with st.spinner("Initializing MCQ Generator..."):
            st.session_state.generator = MCQGenerator()
    
    generator = st.session_state.generator
    
    # Show status
    st.success("✅ Question Database Ready - No API Token Needed")
    
    # Configuration
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📚 Configuration")
        
        track = st.selectbox(
            "Technology Track:",
            options=list(generator.available_tracks.keys()),
            format_func=lambda x: f"{x.upper()} - {generator.available_tracks[x].split('(')[0].strip()}"
        )
        
        col1_1, col1_2 = st.columns(2)
        with col1_1:
            difficulty = st.selectbox("Difficulty:", ["easy", "medium", "hard"], index=1)
        with col1_2:
            count = st.number_input("Questions:", min_value=1, max_value=10, value=5)
    
    with col2:
        st.subheader("📊 Summary")
        
        # Show settings
        st.info(f"**Track:** {track.upper()}")
        st.info(f"**Level:** {difficulty.title()}")
        st.info(f"**Count:** {count}")
        
        # Show mode
        st.success("**Mode:** Question Database")
    
    st.divider()
    
    # Generate button
    if st.button("🚀 Generate Questions", type="primary", use_container_width=True):
        with st.spinner("Generating questions from database..."):
            try:
                questions = generator.generate_questions(track, difficulty, count)
                st.session_state.questions = questions
                st.session_state.generation_info = {
                    'track': track,
                    'difficulty': difficulty, 
                    'count': len(questions),
                    'mode': 'Question Database',
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                st.success(f"✅ Generated {len(questions)} questions!")
                
            except Exception as e:
                st.error(f"❌ Generation failed: {str(e)}")
    
    # Display questions
    if 'questions' in st.session_state and st.session_state.questions:
        st.divider()
        st.header("📝 Generated Questions")
        
        for i, q in enumerate(st.session_state.questions, 1):
            st.subheader(f"Question {i}")
            st.write(f"**{q['text']}**")
            
            # Answer options
            for j, option in enumerate(q['options']):
                option_letter = chr(65 + j)
                if option == q['correct_answer']:
                    st.success(f"✅ **{option_letter}) {option}**")
                else:
                    st.write(f"{option_letter}) {option}")
            
            # Explanation
            with st.expander("💡 Show Explanation"):
                st.info(q['explanation'])
                st.caption(f"Source: {q['generated_by']}")
            
            if i < len(st.session_state.questions):
                st.divider()
        
        # Export
        st.divider()
        if st.button("📥 Export as JSON"):
            json_data = json.dumps(st.session_state.questions, indent=2)
            st.download_button(
                "Download JSON",
                json_data,
                f"mcq_{track}_{difficulty}_{count}.json",
                "application/json"
            )

if __name__ == "__main__":
    main()