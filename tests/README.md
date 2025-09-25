# 🧪 Testing Framework for Slack Knowledge Bot

This directory contains the comprehensive testing framework for the Slack Knowledge Bot, designed to ensure quality, reliability, and performance of the AI-powered knowledge extraction and query response system.

## 📁 Directory Structure

```
tests/
├── conftest.py              # Pytest configuration and fixtures
├── fixtures/                 # Test data and fixtures
│   ├── conversations.py     # Conversation test data
│   ├── queries.py           # Query test data
│   └── ground_truth.py      # Reference responses
├── generators/               # Synthetic data generation
├── evaluators/               # Quality assessment
├── infrastructure/           # Testing infrastructure
│   ├── test_database.py     # Isolated DB setup
│   ├── test_redis.py        # Isolated Redis setup
│   ├── test_openai.py       # Mock OpenAI responses
│   └── test_slack.py        # Mock Slack API
├── monitoring/               # Quality monitoring
├── integration/              # End-to-end tests
└── unit/                     # Unit tests
    ├── test_services/
    ├── test_workers/
    └── test_api/
```

## 🚀 Quick Start

### Prerequisites

1. **Docker and Docker Compose** installed
2. **Python 3.11+** with pip
3. **PostgreSQL** (for local testing)

### Running Tests

#### Option 1: Docker Environment (Recommended)
```bash
# Run all tests in isolated Docker environment
./run_tests.py docker

# Run specific test types
./run_tests.py unit
./run_tests.py integration
./run_tests.py e2e
./run_tests.py quality

# After the run, reports are saved automatically under ./reports
open reports/quality_*.html  # macOS
xdg-open reports/quality_*.html  # Linux
```

#### Option 2: Local Environment
```bash
# Install test dependencies
pip install -r requirements-test.txt

# Run tests locally
./run_tests.py all
./run_tests.py unit --verbose
./run_tests.py specific --path tests/unit/test_services/

# Generate reports manually (if needed)
python -c "from tests.monitoring.reporting import export_json, export_html; export_json('reports/quality_local.json'); export_html('reports/quality_local.html')"
```

## 🔧 Test Configuration

### Environment Variables

The testing framework uses isolated environments with the following configuration:

```bash
# Test Database
DATABASE_URL=postgresql+asyncpg://postgres:admin@localhost:5433/slackbot_test

# Test Redis
REDIS_URL=redis://localhost:6380/15

# Mock API Keys
OPENAI_API_KEY=test_key_12345
SLACK_BOT_TOKEN=xoxb-test-token
```

### Docker Test Environment

The `docker-compose.test.yml` creates isolated test containers:

- **test-app**: Application with test configuration
- **test-db**: PostgreSQL test database (port 5433)
- **test-redis**: Redis test instance (port 6380)

## 📊 Test Categories

### Unit Tests (`-m unit`)
- Test individual components in isolation
- Mock external dependencies
- Fast execution (< 1 second per test)

### Integration Tests (`-m integration`)
- Test component interactions
- Use real database and Redis
- Moderate execution time (1-5 seconds per test)

### End-to-End Tests (`-m e2e`)
- Test complete workflows
- Full system integration
- Longer execution time (5-30 seconds per test)

### Quality Tests (`-m quality`)
- Test AI response quality
- Evaluate knowledge extraction accuracy
- Performance benchmarking

### Robustness & Consistency (Phase 3)
- Adversarial tests: noise/typos, multi-topic, long-context
- Consistency tests: paraphrase variants and small perturbations
- Temporal/no-source guardrails (coming next)

## 🎯 Quality Assessment

### Metrics Evaluated

1. **Factual Accuracy**: Correctness of extracted information
2. **Completeness**: How well responses address queries
3. **Source Attribution**: Proper citation of sources
4. **Actionability**: How actionable responses are for users

### Quality Thresholds

```python
QUALITY_THRESHOLDS = {
    "factual_accuracy": 0.8,
    "completeness": 0.7,
    "source_attribution": 0.8,
    "actionability": 0.75,
    "overall_quality": 0.75
}
```

## 🔍 Test Data

### Conversation Fixtures

Pre-defined conversation types for testing:

- **Technical**: Troubleshooting and technical discussions
- **Decision**: Decision-making conversations
- **Process**: Process definition and workflow discussions
- **Incomplete**: Conversations without clear resolution

### Query Fixtures

Test queries covering:

- **Factual Retrieval**: "What was decided about X?"
- **Process Inquiry**: "How do we do Y?"
- **Attribution Queries**: "Who said Z?"
- **Temporal Queries**: "What happened when?"

### Ground Truth

Reference data for quality evaluation:

- Expected knowledge extractions
- Reference query responses
- Quality scoring criteria
- Evaluation thresholds

## 🛠️ Mock Services

### OpenAI Mock
- Consistent responses for testing
- Configurable response mapping
- Call history tracking
- Embedding generation

### Slack Mock
- Message sending simulation
- User/channel info mocking
- Conversation history simulation
- Message verification utilities

## 📈 Monitoring and Reporting

### Quality Dashboard
- Real-time quality metrics
- Trend analysis
- Threshold monitoring
- Alert generation

### Coverage Reports
### Quality Reports (JSON/HTML)
- Auto-export at end of pytest session to `./reports/quality_<timestamp>.(json|html)`
- Manual exports available via `tests/monitoring/reporting.py`
  - JSON: `export_json('reports/quality.json')`
  - HTML: `export_html('reports/quality.html')`

### Baseline & Diff
- Save a baseline: `from tests.monitoring.reporting import save_baseline; save_baseline('reports/baseline.json')`
- Compare with current: `from tests.monitoring.reporting import diff_against_baseline; print(diff_against_baseline('reports/baseline.json'))`
- Code coverage analysis
- HTML coverage reports
- Coverage thresholds
- Missing coverage identification

## 🚨 Troubleshooting

### Common Issues

1. **Database Connection Errors**
   ```bash
   # Check if test database is running
   docker-compose -f docker-compose.test.yml ps
   
   # Restart test environment
   docker-compose -f docker-compose.test.yml down
   docker-compose -f docker-compose.test.yml up --build
   ```

2. **Redis Connection Errors**
   ```bash
   # Check Redis port
   netstat -an | grep 6380
   
   # Test Redis connection
   redis-cli -p 6380 ping
   ```

3. **Test Failures**
   ```bash
   # Run with verbose output
   ./run_tests.py all --verbose
   
   # Run specific failing test
   ./run_tests.py specific --path tests/unit/test_services/test_slack_service.py
   ```

### Debug Mode

Enable debug logging:

```bash
export LOG_LEVEL=DEBUG
./run_tests.py all --verbose
```

## 📚 Best Practices

### Writing Tests

1. **Use descriptive test names**
2. **Test one thing per test**
3. **Use fixtures for common setup**
4. **Mock external dependencies**
5. **Clean up after tests**

### Test Data

1. **Use realistic test data**
2. **Include edge cases**
3. **Test both positive and negative scenarios**
4. **Maintain test data consistency**

### Quality Assessment

1. **Set appropriate quality thresholds**
2. **Regularly update ground truth data**
3. **Monitor quality trends over time**
4. **Investigate quality regressions immediately**

## 🔄 Continuous Integration

### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### CI Pipeline

The testing framework integrates with CI/CD pipelines:

1. **Code Quality Checks**: Linting, formatting, type checking
2. **Unit Tests**: Fast feedback on code changes
3. **Integration Tests**: Component interaction validation
4. **Quality Tests**: AI response quality assessment
5. **Performance Tests**: System performance validation
 6. **Quality Reports**: Upload `./reports/quality_*.json` and `./reports/quality_*.html` as artifacts

## 📞 Support

For issues with the testing framework:

1. Check the troubleshooting section
2. Review test logs for error details
3. Verify test environment setup
4. Check Docker container status

## 🎯 Next Steps

1. **Phase 2**: Implement quality evaluators
2. **Phase 3**: Add advanced testing capabilities
3. **Phase 4**: Deploy continuous monitoring
4. **Phase 5**: Optimize test performance

---

This testing framework ensures the Slack Knowledge Bot maintains high quality and reliability while enabling rapid development and deployment.
