# nyc-taxi-trips
Welcome! This repo contains a reproducible ML pipeline, with the steps outlined below. The goal is to explore a subset of the NYC taxi trip dataset (~30 million rows) with a deployed machine learning pipeline application for production.

This project aims to emulate key principles of machine learning system architecture: reproducibility of predictions, automated model pipeline, extensibility to add/update models, modular code, scalable to serve predictions to users, and testing.

**Machine learning**  
- [X] Data analysis/EDA
- [X] Feature engineering
- [X] Feature selection
- [X] Train model
- [ ] Data validation
- [ ] Versioning & logging
- [ ] Building a Python package

**Deploy model**
- [ ] Create REST API
- [ ] Add config & logging
- [ ] API schema validation

**CI/CD**
- [ ] Setup & configure CircleCI
- [ ] Publish model to Gemfury
- [ ] Test CI pipeline

**Differential testing**
- [ ] DI

**Deploy to a PaaS**
- [ ] Deploy to Heroku using CI

**Run app with container**
- [ ] Create an API app Dockerfile
- [ ] Build & run a Docker container
- [ ] Release to Heroku utilising Docker

**Deploying to an IaaS**
- [ ] Configure the AWS CLI
- [ ] Elastic container registry
- [ ] Elastic container service
- [ ] Deploying to ECS using CI
