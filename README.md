### Project Title: Copyright in Generative AI Models

### Overview
This project investigates the capabilities of artificial intelligence, specifically utilizing the GPT-3.5 model, in generating children's stories and differentiating them from human-written texts. The project aims to address the creative potentials of AI in literature and its implications on copyright law, focusing on generating unique content and identifying potentially infringing material. The study uses a balanced dataset of human-written and AI-generated stories to establish classification thresholds for distinguishing between the two.

### Authors
- Maria Paula Sanchez (msanc371@fiu.edu)
- Sean Dollinger (sdoll004@fiu.edu)
- Jayavardhan Reddy Samidi (jsami003@fiu.edu)

### Abstract
This project conducts a comparative analysis using metrics such as perplexity, burstiness, lexical diversity, sentence complexity, and cosine similarity. By creating and analyzing a balanced dataset of human-written and AI-generated stories, the project establishes classification thresholds that accurately distinguish between the two, offering insights into AI's role in content creation and its challenges in the digital copyright landscape.

### CCS Concepts
- Tokenization
- TF-IDF
- Cosine Similarity

### Keywords
- Copyright
- Perplexity
- Burstiness
- Sentence Complexity
- Lexical Diversity
- GPT-2
- Artificial Intelligence (AI)
- GPT-3
- ChatGPT

### Introduction
Artificial intelligence presents new potential and challenges for copyright law, especially in literary works. This project aims to determine how much AI can generate original children's stories and establish a methodology for differentiating them from human-written texts. The study focuses on using various linguistic features to evaluate the level of similarity between these children's stories.

### Method Design

#### Overview of the System
The system uses a Streamlit web app to check text closely using important language metrics such as sentence complexity, lexical diversity, and text flow. The main aim is to understand the text's originality and ensure copyright compliance.

#### Choice of Technologies
- **Streamlit**: For its ease of use and immediate metric updates.
- **Transformers and PyTorch**: For processing language and analyzing text perplexity.
- **NLTK**: For text processing tasks like tokenization and stop-word removal.
- **Scikit-learn**: For TF-IDF vectorization and cosine similarity analysis.

### Text Preprocessing

#### Regex Operations
Used for cleaning and standardizing the text by removing unnecessary characters.

#### Tokenization and Stopwords Removal
Breaking down text into individual words or tokens and removing common words that do not contribute much information to the text analysis.

#### Normalization
Converting all text into a uniform format, such as lowercase, to prevent inconsistencies.

### Analytical Techniques

#### TF-IDF and Cosine Similarity
Evaluates the importance of words in a document relative to a larger corpus and measures the degree of similarity between two text contents.

#### Perplexity
Measures how well a probability model predicts a sample, indicating the complexity of the text.

#### Burstiness
Describes the variance in the frequency of words in the text, indicating repetitive content.

#### Lexical Diversity
Measures the variety of words in text content.

#### Sentence Complexity
Analyzes the structure of sentences within a document, providing insights into the syntax of the text.

### Implementation

#### AI Model Overview
The central part of the project uses the GPT-2 model to analyze text content. Each measure was selected to meet the requirement of copyright detection with the help of the GPT-2 model.

#### Structure and Functions of the Project Code
- **Text Preprocessing Module**: Standardizes and cleanses text data.
- **Metric Calculation Module**: Calculates sentence complexity, lexical diversity, similarity score, burstiness, and perplexity.
- **Analysis Engine**: Coordinates the preparation module and metric computations.
- **User Interface**: Streamlit-built interface for entering text and viewing analysis findings instantly.

### Threshold Values for Metrics
- **Perplexity**: Greater than 30,000
- **Burstiness**: Less than 0.25
- **Lexical Diversity**: Less than 0.71
- **Sentence Complexity**: Between 15 and 25

### Results and Discussion

#### Perplexity Analysis
AI-generated texts show higher average perplexity compared to the original texts, indicating a higher degree of novelty.

#### Burstiness vs. Lexical Diversity
Negative correlation between burstiness and lexical diversity is more pronounced in original texts than in AI-generated ones, suggesting a difference in language use.

#### Sentence Complexity
No significant difference in sentence complexity between AI-generated and original texts, indicating effective replication of human sentence structures by AI.

#### Similarity Scores
Most AI-generated texts have a moderate degree of similarity to original texts, suggesting that AI is not simply copying but also not creating highly distinct content.

#### Cross-Metric Correlation Analysis
Strong correlations between certain metrics suggest underlying patterns in how AI generates text, helping predict and control factors that contribute to originality and diversity.

### Conclusion
The AI model demonstrates the potential for creating original content but requires careful monitoring to minimize the risk of copyright infringement. The project contributes to understanding AI-generated literature's originality and proposes a Copyright Alerting System bolstered by human review to ensure copyright compliance effectively.

### Contact
For further information or questions, please contact the project maintainers at the provided email addresses.
