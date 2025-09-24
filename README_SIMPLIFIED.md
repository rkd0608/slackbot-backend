# Slack Knowledge Bot - Simplified Architecture

**A focused team knowledge assistant that makes your Slack conversations searchable and actionable.**

## 🎯 **Core Value Proposition**

**"We make your team's specific conversations and decisions searchable and actionable"** - not general AI reasoning.

Your only job is helping teams find and understand their own institutional knowledge. This system focuses on:
- Extract decisions from team conversations with proper context
- Surface step-by-step processes team members have shared  
- Find relevant past discussions when similar topics come up
- Track who said what and when for accountability

## 🏗️ **Simplified Architecture**

### **Core Services (Essential Only)**
- `simple_message_processor.py` - Process incoming Slack messages
- `simple_knowledge_extractor.py` - Extract knowledge from complete conversations  
- `simple_query_processor.py` - Handle user queries and generate responses
- `vector_service.py` - Find relevant knowledge for queries
- `slack_service.py` - Slack API communication

### **Eliminated Over-Engineering**
❌ **Removed**: `advanced_ai_intelligence.py` - Over-engineered ChatGPT competitor  
❌ **Removed**: `conversational_intelligence.py` - Redundant with query processor  
❌ **Removed**: `direct_conversation_analyzer.py` - Should be part of knowledge extractor  
❌ **Removed**: `context_resolver.py` - Should be part of query processor  
❌ **Removed**: `intent_classifier.py` - Unnecessary complexity  
❌ **Removed**: `process_recognizer.py` - Should be part of knowledge extractor  
❌ **Removed**: `conversation_state_manager.py` - Over-engineered  
❌ **Removed**: `realtime_knowledge_extractor.py` - Contradicts batch architecture

## 🔄 **Single Processing Pipeline**

### **Step 1: Message Ingestion**
```
Slack Message → simple_message_processor → Database Storage
```

### **Step 2: Knowledge Extraction** 
```
Completed Conversations → simple_knowledge_extractor → Knowledge Items with Embeddings
```

### **Step 3: Query Processing**
```
User Query → simple_query_processor → Search Knowledge → Generate Response → Slack
```

## 📊 **Simplified Database Schema**

### **Essential Tables Only**
- `workspaces` - Customer workspaces
- `users` - Slack users  
- `messages` - Raw Slack messages with threading info
- `conversations` - Message groupings
- `knowledge_items` - Extracted knowledge with embeddings
- `queries` - User queries and responses for learning

## 🤖 **AI Processing Strategy**

### **Cost-Effective Model Usage**
- **GPT-3.5-turbo** for knowledge extraction (90% cheaper than GPT-4)
- **GPT-4** only for complex queries that really need it (rarely)
- **Batch processing** conversations to reduce API costs
- **Cache** common queries to avoid re-processing

### **Single Response Format**
```
[Emoji] **Clear Answer to User's Question**

[Specific information from team conversations]

**Source**: [Who said it] on [when] in [#channel]
**Participants**: [@person1] [@person2]
```

## 🚀 **Getting Started**

### **Prerequisites**
- Docker & Docker Compose
- PostgreSQL with pgvector extension
- Redis for task queue
- OpenAI API key
- Slack App with Bot Token

### **Quick Start**
```bash
# Clone and setup
git clone <repository>
cd slackbot-backend

# Environment setup
cp .env.example .env
# Edit .env with your keys

# Start services
docker-compose up -d

# Check status
docker-compose logs --tail=20
```

### **Test the System**
```bash
curl -X POST http://localhost:8000/api/v1/slack/commands/ask \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "token=test&team_id=T123&channel_id=C123&user_id=U123&text=what is docker?"
```

## 🔧 **Configuration**

### **Essential Environment Variables**
```env
# Database
DATABASE_URL=postgresql://postgres:admin@db:5432/slackbot

# Redis
REDIS_URL=redis://redis:6379/0

# OpenAI
OPENAI_API_KEY=sk-...

# Slack
SLACK_CLIENT_ID=123...
SLACK_CLIENT_SECRET=abc...
SLACK_SIGNING_SECRET=xyz...
```

## 🏃‍♂️ **Development Workflow**

### **Simplified Workers**
```bash
# Query worker (high priority)
docker-compose logs -f celery-query-worker

# Background worker (knowledge extraction)  
docker-compose logs -f celery-background-worker
```

### **Testing Queries**
```bash
# Test basic functionality
curl -X POST http://localhost:8000/api/v1/slack/commands/ask \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "token=test&team_id=T123&channel_id=C123&user_id=U123&text=your question here"
```

## 📈 **Success Metrics**

### **Technical Success**
- ✅ Users can reliably find past team decisions
- ✅ Responses include specific details and proper attribution  
- ✅ System works without breaking or giving wrong information
- ✅ Response time under 10 seconds (not 3 seconds)

### **Product Success**
- ✅ Teams use it multiple times per week
- ✅ Users prefer it to searching Slack directly
- ✅ Clear evidence it saves time on finding team information
- ✅ Organic word-of-mouth referrals from satisfied users

### **Business Success**  
- ✅ Sustainable unit economics with realistic AI costs
- ✅ Clear path to profitability at reasonable scale
- ✅ Customers willing to pay for proven value
- ✅ Market validation for team knowledge management need

## 🎯 **Target Market**

### **Focus Areas**
- **Engineering teams** (10-50 people)
- **Who make frequent technical decisions**
- **Who struggle to find past decisions and processes**  
- **Who value accuracy over speed**

### **Value Proposition**
**"Never lose important team decisions and processes again"** - not "AI reasoning assistant for teams."

## 🚫 **What We DON'T Do**

- ❌ General AI reasoning or explanations
- ❌ Compete with ChatGPT/Claude capabilities  
- ❌ Multi-modal reasoning systems
- ❌ Real-time analysis promises
- ❌ Complex query routing
- ❌ Advanced reasoning capabilities

## 📝 **API Endpoints**

### **Essential Endpoints**
```
POST /api/v1/slack/commands/ask    # Main slash command
POST /api/v1/slack/events          # Slack events webhook  
GET  /api/v1/health                # Health check
```

## 🔍 **Monitoring & Debugging**

### **Key Logs to Watch**
```bash
# Query processing
docker-compose logs -f celery-query-worker | grep "Processing query"

# Knowledge extraction  
docker-compose logs -f celery-background-worker | grep "knowledge"

# API responses
docker-compose logs -f app | grep "ask"
```

### **Common Issues**
1. **No responses**: Check worker logs for errors
2. **Empty knowledge**: Verify conversations are being processed
3. **Slow responses**: Check OpenAI API usage and rate limits

## 📚 **Knowledge Extraction Process**

### **Conversation Assembly**
- Group related messages into conversation threads
- Wait for conversation completion (no new messages for 10 minutes)  
- Assemble full context including participants and timeline

### **Knowledge Extraction**
- Use one well-tuned prompt for extraction
- Focus on extracting specific, actionable information
- Include proper source attribution and confidence scoring

### **Storage and Indexing**  
- Store extracted knowledge with vector embeddings
- Index for fast retrieval by topic, participant, and time

## 🔄 **Deployment**

### **Production Setup**
```bash
# Production environment
docker-compose -f docker-compose.prod.yml up -d

# Scale workers based on load
docker-compose up --scale celery-query-worker=2
```

### **Monitoring**
- Monitor query response times
- Track knowledge extraction success rates
- Watch API costs and usage patterns
- Monitor user engagement metrics

---

## 📞 **Support**

For issues or questions:
1. Check logs first: `docker-compose logs`
2. Review this README for common solutions
3. Focus on the core functionality - keep it simple

**Remember**: This is a team knowledge assistant, not a general AI system. Keep the focus narrow and execution excellent.
