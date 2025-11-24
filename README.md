# Smart Resume Screening AI Agent Chain

🔬 **Intelligent HR Recruitment System powered by AI agent orchestration**

An advanced Python-based system that automates resume screening and candidate evaluation using a sophisticated 5-agent AI pipeline. Designed for HR departments and recruiters to efficiently process candidates at scale while ensuring fair, explainable, and comprehensive evaluations.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-0.1.0-orange.svg)](https://www.langchain.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-API-green.svg)](https://openai.com/)

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key-here"

# Screen a single resume
python main.py screen tests/sample_data/sample_resume.txt tests/sample_data/sample_job_description.json --output result.json

# Batch process multiple resumes
python main.py batch data/resumes/ tests/sample_data/sample_job_description.json --output batch_report.json
```

## 🏗️ System Architecture

### The 5-Agent AI Pipeline

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Document      │    │   Information   │    │   Job Matcher   │    │    Scorer &     │    │   Report        │
│   Parser        │    │   Extractor     │    │   & Analyzer    │    │    Ranker       │    │   Generator     │
│   (Agent 1)     │    │   (Agent 2)     │    │   (Agent 3)     │    │   (Agent 4)     │    │   (Agent 5)     │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │                       │                       │
         ▼                       ▼                       ▼                       ▼                       ▼
   Clean Text ─────────────►  Structured Data ────────────► Match Analysis ───────────► Scores & Recommendations ─────► Final Reports

                                                              Semantic
                                                          Skills Matching
                                                             & Vector Search
```

### Agent Responsibilities

1. **📄 Document Parser** - Cleans and formats raw resume text from multiple file formats (PDF, DOCX, TXT, HTML)
2. **🔍 Information Extractor** - Extracts structured data: contact info, skills, experience, education, certifications
3. **🎯 Job Matcher** - Semantic matching using vector search to analyze candidate fit against job requirements
4. **📊 Scorer & Ranker** - Weighted evaluation with numerical scoring (0-100) and hiring recommendations
5. **📋 Report Generator** - Creates comprehensive executive reports with interview questions and ranking

## 📋 Key Features

### Multi-Format Resume Processing
- ✅ PDF documents (with OCR support via PyPDF2)
- ✅ Microsoft Word (.docx) files
- ✅ Plain text (.txt) files
- ✅ HTML resume formats

### Intelligent Skill Matching
- 🔍 Semantic similarity search using sentence transformers
- 🎯 Vector-based comparison of technical/soft skills
- 📈 Configurable similarity thresholds

### Comprehensive Evaluation
- 📊 Weighted scoring system (configurable weights)
- 🧠 AI-powered recommendations: STRONG HIRE → HIRE → MAYBE → PASS
- 💪 Strengths and weaknesses analysis
- 🎯 Skills gaps identification
- 💬 Tailored interview questions generation

### Enterprise-Ready Features
- ⚡ Batch processing for high-volume recruitment
- 🔒 Privacy-focused (no external data storage)
- 📈 Scalable architecture with LangGraph
- 🔧 Configurable scoring thresholds and weights
- 📝 Detailed audit trails and reasoning

### Report Formats
- 📄 JSON reports (detailed analysis)
- 📊 CSV summaries (for bulk operations)
- 📈 Ranked candidate lists with recommendations
- 💬 Auto-generated interview questions

## 🛠️ Installation

### Prerequisites
- Python 3.10+
- OpenAI API key (or compatible LLM API)

### Setup
```bash
# Clone the repository
git clone https://github.com/your-org/ai_assistant_hr.git
cd ai_assistant_hr

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Configuration

Customize behavior via `config.yaml`:

```yaml
scoring:
  weights:
    skills_match: 0.40
    experience_relevance: 0.30
    education_match: 0.15
    cultural_fit: 0.15

matching:
  min_similarity_score: 0.6
  required_skills_weight: 1.5

agents:
  timeout_seconds: 120
  max_retries: 3
```

## 📖 Usage Examples

### Single Resume Screening
```bash
# Basic screening with pretty output
python main.py screen resume.pdf job_description.json

# Save detailed JSON report
python main.py screen resume.pdf job_description.json --output evaluation_report.json

# Custom configuration
python main.py screen resume.pdf job_description.json --config custom_config.yaml --verbose
```

### Batch Processing
```bash
# Process all resumes in a directory
python main.py batch resumes/ job_description.json --output batch_results.json --max-resumes 50

# Generate CSV summary report
python main.py batch resumes/ job_description.json --output summary.csv --format csv
```

### Utility Commands
```bash
# Validate resume files
python main.py validate --directory data/resumes/

# Parse and structure job description
python main.py parse-job job_posting.txt --output structured_job.json
```

### Python API Usage
```python
from core.chain_orchestrator import ResumeScreeningChain, ChainConfig

# Initialize the system
config = ChainConfig.from_yaml("config.yaml")
chain = ResumeScreeningChain(config)

# Load job description
import json
with open("job_desc.json", "r") as f:
    job_desc = json.load(f)

# Process single resume
result = chain.process_resume("resume.pdf", job_desc, "output.json")

# Process multiple resumes
results = chain.process_batch_resumes(["resume1.pdf", "resume2.pdf"], job_desc, "batch_report.json")
```

## 📊 Sample Output

### Single Candidate Report
```
📊 Overall Assessment
Score: 92/100
Recommendation: STRONG HIRE

📈 Score Breakdown
Component              Score     Weight
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Skills Match           95        40%
Experience Relevance   90        30%
Education Match        85        15%
Cultural Fit           88        15%

💪 Key Strengths
• Expert in Python and Django frameworks
• Strong leadership and mentoring experience
• Proven track record with high-traffic applications
• Excellent cloud architecture skills

⚠️ Concerns
• Limited exposure to newer JavaScript frameworks

Suggested Interview Questions:
1. How have you approached architecting microservices infrastructure in your previous roles?
2. Can you describe your experience mentoring junior developers and improving team processes?
3. Tell me about a challenging technical decision you made regarding cloud infrastructure.
4. How do you stay current with Python ecosystem developments?
```

### Batch Processing Results
```
🏆 Top Candidates Summary
Rank  Name              Score  Recommendation    Key Highlights
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1     Sarah Johnson     96     STRONG HIRE       Python expert, 8+ years exp, AWS certified
2     Michael Chen      89     STRONG HIRE       Full-stack development, technical leadership
3     Emily Rodriguez   87     HIRE              Senior dev, strong Django background
4     David Park        82     HIRE              6 years exp, excellent communication
5     Lisa Wong         79     MAYBE             Good skills, needs more experience
```

## 🔧 Customization

### Modifying Scoring Weights
Edit `config.yaml` to adjust evaluation criteria:

```yaml
scoring:
  weights:
    skills_match: 0.50          # Increase importance of technical skills
    experience_relevance: 0.25
    education_match: 0.10
    cultural_fit: 0.15
```

### Adding New Agent Capabilities
Extend agents by modifying the prompt templates in `utils/prompt_templates.py`:

```python
NEW_AGENT_PROMPT = """Your custom prompt for specialized evaluation..."""
```

### Integrating Different LLMs
The system is designed to work with various LLM providers:

```python
# For Anthropic Claude
llm_config = {"model": "claude-3-sonnet-20240229", "api_key": "your-anthropic-key"}
```

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test
pytest tests/test_basic.py::test_file_handler_validation -v

# Test with sample data
python main.py validate --path tests/sample_data/sample_resume.txt
```

## 🔒 Privacy & Compliance

- **Local Processing**: All analysis performed locally, no candidate data transmitted externally
- **Configurable Retention**: Automated cleanup policies for temporary files
- **Bias Mitigation**: Transparent scoring formulas with configurable weights
- **Audit Trails**: Complete logging of evaluation reasoning and decision factors
- **GDPR Compliant**: No external data storage or third-party sharing

## 📈 Performance Considerations

- **Batch Processing**: Designed for processing 50+ resumes simultaneously
- **LLM Optimization**: Configurable token limits and retry logic
- **Vector Search**: Efficient semantic matching for skill comparison
- **Memory Management**: Streaming processing for large resume files

## 🐛 Troubleshooting

### Common Issues

**API Rate Limits**
- Reduce batch size: `--max-resumes 10`
- Add delays between requests in config
- Consider upgrading API plan

**Memory Issues**
- Process resumes individually for large files
- Check file size limits in config
- Monitor system resources during batch operations

**Low-Quality Results**
- Verify resume file quality (OCR for scanned PDFs)
- Adjust similarity thresholds in config
- Review job description structure

### Getting Help

1. Check logs with `--verbose` flag
2. Validate files: `python main.py validate --path your_file.pdf`
3. Test with sample data: `python main.py screen --help`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

### Development Setup
```bash
pip install -r requirements.txt
pip install -r requirements-dev.txt  # if testing dependencies exist
pre-commit install  # if pre-commit hooks configured
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built with [LangChain](https://www.langchain.com/) and [LangGraph](https://github.com/langchain-ai/langgraph)
- Semantic search powered by [Sentence Transformers](https://www.sbert.net/)
- Inspired by modern AI agent orchestration patterns

## 📞 Support

For questions, issues, or feature requests:
- 📧 Open an [issue](https://github.com/your-org/ai_assistant_hr/issues) on GitHub
- 📖 Check the [documentation](https://github.com/your-org/ai_assistant_hr/wiki)
- 💬 Join our [discussion forum](https://github.com/your-org/ai_assistant_hr/discussions)

---

**📈 Ready to transform your recruitment process with AI? Start automating candidate evaluation today!**
