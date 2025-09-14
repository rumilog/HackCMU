# 🦟 Lantern Fly Tracker

A mobile-friendly web application that uses computer vision to identify dead lantern flies from user photos, tracks their locations, and gamifies the process with a points system.

## 🎯 Features

- **Mobile-First Design**: Progressive Web App (PWA) optimized for mobile devices
- **Photo Classification**: AI-powered detection of dead lantern flies using computer vision
- **Geolocation Tracking**: Automatic capture and storage of photo locations
- **Gamification**: Points system and achievements for user engagement
- **Interactive Map**: Visual representation of reported lantern fly locations
- **User Authentication**: Secure user accounts with JWT authentication
- **Real-time Updates**: Live leaderboard and statistics

## 🏗️ Architecture

```
hackCMU/
├── frontend/          # React TypeScript web app
├── backend/           # Node.js Express API with SQLite database
├── ml-model/          # Python ML model and inference
├── database/          # SQLite schemas and migrations
└── deployment/        # Docker and deployment configs
```

## 🚀 Quick Start

### Prerequisites

- Node.js 18+
- Python 3.11+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd hackCMU
   ```

2. **Install all dependencies**
   ```bash
   npm run setup
   ```

3. **Set up environment variables**
   ```bash
   # Backend
   cp backend/env.example backend/.env
   # Edit backend/.env with your configuration
   ```

4. **Start the development servers**

   **Option A: Automated Start (Recommended)**
   ```bash
   python start_ml_integration.py
   ```

   **Option B: Manual Start**
   ```bash
   # Terminal 1 - Start ML Service
   cd ml-model
   python inference/api_server.py
   
   # Terminal 2 - Start Backend
   cd backend
   npm run dev
   
   # Terminal 3 - Start Frontend
   cd frontend
   npm run dev
   ```

   **Option C: PowerShell (Windows)**
   ```powershell
   .\start_backend.ps1
   ```

This will start:
- Frontend: http://localhost:3000
- Backend API: http://localhost:5000
- ML Model API: http://localhost:5001

## 🧪 Testing

Test the ML integration:

```bash
# Test ML service directly
python test_ml_integration.py

# Test classification
python test_classification.py
```

## 🐳 Docker Deployment

For production deployment using Docker:

```bash
cd deployment/docker
docker-compose up -d
```

## 📱 Mobile Usage

The application is designed as a Progressive Web App (PWA) and can be installed on mobile devices:

1. Open the app in your mobile browser
2. Look for the "Add to Home Screen" option
3. The app will work like a native mobile app

## 🔧 Development

### Frontend Development

```bash
cd frontend
npm run dev
```

### Backend Development

```bash
cd backend
npm run dev
```

### ML Model Development

```bash
cd ml-model
pip install -r inference/requirements.txt
python inference/api_server.py
```

## 📊 Database Schema

The application uses SQLite for data storage:

- **users**: User accounts and statistics
- **photos**: Photo metadata and classification results
- **photo_embeddings**: ML embeddings for future duplication detection
- **achievements**: User achievements and rewards

## 🤖 Machine Learning

The ML pipeline includes:

- **Image Classification**: EfficientNet-B0 model for lantern fly detection
- **Data Augmentation**: Enhanced training with various image transformations
- **Model Serving**: Flask-based inference service on port 5001
- **Model Performance**: 99.43% validation accuracy, 89.7% test accuracy
- **Future Features**: Duplication detection using image embeddings

## 🎮 Gamification

- **Points System**: Earn points for photo uploads and confirmed lantern flies
- **Achievements**: Unlock achievements for various milestones
- **Leaderboard**: Compete with other users
- **Accuracy Tracking**: Monitor your classification accuracy

## 🌍 Environmental Impact

This application helps researchers and environmentalists:

- Track lantern fly populations
- Identify infestation hotspots
- Contribute to scientific research
- Raise awareness about invasive species

## 📈 Future Enhancements

- [ ] Duplication detection using image similarity
- [ ] Advanced analytics and heat maps
- [ ] Community features and photo sharing
- [ ] Native mobile apps
- [ ] Integration with research databases
- [ ] Real-time notifications
- [ ] Batch photo upload

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
- Create an issue in the repository
- Contact the development team
- Check the documentation

## 🙏 Acknowledgments

- CMU Hackathon organizers
- Environmental research community
- Open source contributors
- Machine learning community

---

**Made with ❤️ for environmental conservation**
