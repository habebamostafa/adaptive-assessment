require("dotenv").config({ path: require('path').resolve(__dirname, '.env') });

const express = require('express');
const mongoose = require('mongoose');
const jwt = require('jsonwebtoken');
const bcrypt = require('bcryptjs');
const { spawn } = require('child_process');
const path = require('path');
const cors = require('cors');

const app = express();

// Middleware
app.use(cors());
app.use(express.json());

// Debug: Check if environment variables are loaded
console.log('MONGODB_URI:', process.env.MONGO_URI);
console.log('JWT_SECRET:', process.env.JWT_SECRET ? 'Loaded' : 'Not loaded');
console.log('PORT:', process.env.PORT);

// MongoDB Connection with error handling
const MONGO_URI = process.env.MONGO_URI || "mongodb://localhost:27017/adaptive-assessment";

mongoose.connect(MONGO_URI)
.then(() => console.log('✅ Connected to MongoDB'))
.catch((error) => {
  console.error('❌ MongoDB connection error:', error.message);
  console.log('💡 Please make sure MongoDB is running on your system');
  console.log('   You can install it from: https://www.mongodb.com/try/download/community');
  console.log('   Or use MongoDB Atlas: https://www.mongodb.com/atlas');
});

// Enhanced User Schema with RL Integration
const userSchema = new mongoose.Schema({
  name: { type: String, required: true },
  email: { type: String, required: true, unique: true },
  password: { type: String, required: true },
  role: { type: String, enum: ['student', 'teacher'], default: 'student' },
  learningProfile: {
    preferredLearningStyle: { type: String, enum: ['visual', 'auditory', 'kinesthetic', 'reading'], default: 'visual' },
    difficultyLevel: { type: String, enum: ['beginner', 'intermediate', 'advanced'], default: 'beginner' },
    track: { type: String, enum: ['web', 'ai', 'cyber', 'data', 'mobile', 'devops'], default: 'web' },
    
    // RL Assessment Data
    assessmentHistory: [{
      sessionId: String,
      track: String,
      startTime: { type: Date, default: Date.now },
      endTime: Date,
      totalQuestions: Number,
      correctAnswers: Number,
      finalAbility: Number,
      confidenceScore: Number,
      recommendedLevel: Number,
      questionHistory: [{
        questionId: String,
        level: Number,
        question: String,
        selectedAnswer: String,
        correctAnswer: String,
        isCorrect: Boolean,
        abilityAfter: Number,
        timestamp: { type: Date, default: Date.now }
      }],
      agentMetrics: {
        agentType: String,
        adaptationStrategy: String,
        qTableSummary: mongoose.Schema.Types.Mixed
      }
    }],
    
    // Adaptive Learning Metrics
    currentAbility: { type: Number, default: 0.5 },
    learningVelocity: { type: Number, default: 0 },
    preferredDifficulty: { type: Number, default: 1 },
    masteredTopics: [String],
    strugglingTopics: [String],
    
    completedLessons: [{ type: mongoose.Schema.Types.ObjectId, ref: 'Lesson' }],
    scores: [{
      lessonId: { type: mongoose.Schema.Types.ObjectId, ref: 'Lesson' },
      score: Number,
      attempts: Number,
      completedAt: { type: Date, default: Date.now }
    }]
  }
}, { timestamps: true });

// Assessment Session Schema
const assessmentSessionSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User', required: true },
  track: { type: String, required: true },
  status: { type: String, enum: ['active', 'completed', 'abandoned'], default: 'active' },
  
  // RL Environment State
  studentAbility: { type: Number, default: 0.5 },
  currentLevel: { type: Number, default: 2 },
  confidenceScore: { type: Number, default: 0 },
  totalQuestionsAsked: { type: Number, default: 0 },
  maxQuestions: { type: Number, default: 10 },
  confidenceThreshold: { type: Number, default: 0.8 },
  
  // Agent Configuration
  agentType: { type: String, default: 'main' },
  adaptationStrategy: { type: String, default: 'rl_based' },
  
  // Session Data
  questionHistory: [{
    questionId: String,
    level: Number,
    questionText: String,
    options: [String],
    selectedAnswer: String,
    correctAnswer: String,
    isCorrect: Boolean,
    abilityBefore: Number,
    abilityAfter: Number,
    timestamp: { type: Date, default: Date.now }
  }],
  
  performanceHistory: [{
    questionNumber: Number,
    ability: Number,
    confidence: Number,
    level: Number
  }]
}, { timestamps: true });

// Models
const User = mongoose.model('User', userSchema);
const AssessmentSession = mongoose.model('AssessmentSession', assessmentSessionSchema);

// JWT Middleware
const authenticateToken = (req, res, next) => {
  const authHeader = req.headers['authorization'];
  const token = authHeader && authHeader.split(' ')[1];

  if (!token) {
    return res.status(401).json({ message: 'Access token required' });
  }

  jwt.verify(token, process.env.JWT_SECRET, (err, user) => {
    if (err) return res.status(403).json({ message: 'Invalid token' });
    req.user = user;
    next();
  });
};

// Helper function to call Python RL system
const callRLSystem = (action, data) => {
  return new Promise((resolve, reject) => {
    const pythonProcess = spawn('python', [
      path.join(__dirname, 'rl_adapter.py'),
      action,
      JSON.stringify(data)
    ]);

    let result = '';
    let error = '';

    pythonProcess.stdout.on('data', (data) => {
      result += data.toString();
    });

    pythonProcess.stderr.on('data', (data) => {
      error += data.toString();
    });

    pythonProcess.on('close', (code) => {
      if (code === 0) {
        try {
          resolve(JSON.parse(result));
        } catch (e) {
          reject(new Error('Invalid JSON response from RL system'));
        }
      } else {
        reject(new Error(`RL system error: ${error}`));
      }
    });
  });
};

// Auth Routes (same as before)
app.post('/api/register', async (req, res) => {
  try {
    const { name, email, password, role, preferredLearningStyle, track } = req.body;
    
    const existingUser = await User.findOne({ email });
    if (existingUser) {
      return res.status(400).json({ message: 'User already exists' });
    }

    const hashedPassword = await bcrypt.hash(password, 10);
    
    const user = new User({
      name,
      email,
      password: hashedPassword,
      role: role || 'student',
      learningProfile: {
        preferredLearningStyle: preferredLearningStyle || 'visual',
        track: track || 'web'
      }
    });

    await user.save();

    const token = jwt.sign(
      { userId: user._id, email: user.email, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    res.status(201).json({
      message: 'User created successfully',
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        learningProfile: user.learningProfile
      }
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

app.post('/api/login', async (req, res) => {
  try {
    const { email, password } = req.body;

    const user = await User.findOne({ email });
    if (!user) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    const isMatch = await bcrypt.compare(password, user.password);
    if (!isMatch) {
      return res.status(400).json({ message: 'Invalid credentials' });
    }

    const token = jwt.sign(
      { userId: user._id, email: user.email, role: user.role },
      process.env.JWT_SECRET,
      { expiresIn: '24h' }
    );

    res.json({
      message: 'Login successful',
      token,
      user: {
        id: user._id,
        name: user.name,
        email: user.email,
        role: user.role,
        learningProfile: user.learningProfile
      }
    });
  } catch (error) {
    res.status(500).json({ message: 'Server error', error: error.message });
  }
});

// RL Assessment Routes

// Start new assessment session
app.post('/api/assessment/start', authenticateToken, async (req, res) => {
  try {
    const { track, maxQuestions, confidenceThreshold, agentType, adaptationStrategy } = req.body;
    const userId = req.user.userId;

    // Create new assessment session
    const session = new AssessmentSession({
      userId,
      track,
      maxQuestions: maxQuestions || 10,
      confidenceThreshold: confidenceThreshold || 0.8,
      agentType: agentType || 'main',
      adaptationStrategy: adaptationStrategy || 'rl_based'
    });

    await session.save();

    // Initialize RL environment via Python
    const rlResponse = await callRLSystem('init_environment', {
      track,
      maxQuestions,
      confidenceThreshold,
      agentType,
      adaptationStrategy
    });

    // Get first question
    const firstQuestion = await callRLSystem('get_question', {
      sessionId: session._id,
      currentState: {
        ability: session.studentAbility,
        level: session.currentLevel,
        questionsAsked: session.totalQuestionsAsked
      }
    });

    res.json({
      sessionId: session._id,
      question: firstQuestion,
      session: {
        studentAbility: session.studentAbility,
        currentLevel: session.currentLevel,
        totalQuestionsAsked: session.totalQuestionsAsked,
        maxQuestions: session.maxQuestions
      }
    });

  } catch (error) {
    console.error('Assessment start error:', error);
    res.status(500).json({ message: 'Failed to start assessment', error: error.message });
  }
});

// Submit answer and get next question
app.post('/api/assessment/answer', authenticateToken, async (req, res) => {
  try {
    const { sessionId, questionId, selectedAnswer, questionData } = req.body;
    const userId = req.user.userId;

    // Find assessment session
    const session = await AssessmentSession.findOne({ _id: sessionId, userId });
    if (!session) {
      return res.status(404).json({ message: 'Assessment session not found' });
    }

    // Process answer via RL system
    const rlResponse = await callRLSystem('submit_answer', {
      sessionId,
      questionId,
      selectedAnswer,
      questionData,
      currentState: {
        ability: session.studentAbility,
        level: session.currentLevel,
        questionsAsked: session.totalQuestionsAsked
      }
    });

    // Update session in database
    session.questionHistory.push({
      questionId,
      level: questionData.level,
      questionText: questionData.text,
      options: questionData.options,
      selectedAnswer,
      correctAnswer: questionData.correct_answer,
      isCorrect: rlResponse.isCorrect,
      abilityBefore: session.studentAbility,
      abilityAfter: rlResponse.newAbility
    });

    session.performanceHistory.push({
      questionNumber: session.totalQuestionsAsked + 1,
      ability: rlResponse.newAbility,
      confidence: rlResponse.confidence,
      level: rlResponse.newLevel
    });

    // Update session state
    session.studentAbility = rlResponse.newAbility;
    session.currentLevel = rlResponse.newLevel;
    session.confidenceScore = rlResponse.confidence;
    session.totalQuestionsAsked += 1;

    // Check if assessment is complete
    if (rlResponse.done || session.totalQuestionsAsked >= session.maxQuestions) {
      session.status = 'completed';
      
      // Update user's learning profile
      const user = await User.findById(userId);
      user.learningProfile.currentAbility = rlResponse.newAbility;
      user.learningProfile.preferredDifficulty = rlResponse.recommendedLevel;
      
      // Add to assessment history
      user.learningProfile.assessmentHistory.push({
        sessionId: session._id,
        track: session.track,
        endTime: new Date(),
        totalQuestions: session.totalQuestionsAsked,
        correctAnswers: session.questionHistory.filter(q => q.isCorrect).length,
        finalAbility: rlResponse.newAbility,
        confidenceScore: rlResponse.confidence,
        recommendedLevel: rlResponse.recommendedLevel,
        questionHistory: session.questionHistory,
        agentMetrics: rlResponse.agentMetrics
      });

      await user.save();
      await session.save();

      res.json({
        isCorrect: rlResponse.isCorrect,
        explanation: rlResponse.explanation,
        done: true,
        summary: {
          totalQuestions: session.totalQuestionsAsked,
          correctAnswers: session.questionHistory.filter(q => q.isCorrect).length,
          finalAbility: rlResponse.newAbility,
          confidenceScore: rlResponse.confidence,
          recommendedLevel: rlResponse.recommendedLevel,
          levelPerformance: rlResponse.levelPerformance
        }
      });
    } else {
      // Get next question
      const nextQuestion = await callRLSystem('get_next_question', {
        sessionId,
        newState: {
          ability: rlResponse.newAbility,
          level: rlResponse.newLevel,
          questionsAsked: session.totalQuestionsAsked
        }
      });

      await session.save();

      res.json({
        isCorrect: rlResponse.isCorrect,
        explanation: rlResponse.explanation,
        done: false,
        nextQuestion: nextQuestion,
        session: {
          studentAbility: session.studentAbility,
          currentLevel: session.currentLevel,
          confidenceScore: session.confidenceScore,
          totalQuestionsAsked: session.totalQuestionsAsked
        }
      });
    }

  } catch (error) {
    console.error('Answer submission error:', error);
    res.status(500).json({ message: 'Failed to process answer', error: error.message });
  }
});

// Get assessment analytics
app.get('/api/assessment/analytics/:sessionId', authenticateToken, async (req, res) => {
  try {
    const { sessionId } = req.params;
    const userId = req.user.userId;

    const session = await AssessmentSession.findOne({ _id: sessionId, userId });
    if (!session) {
      return res.status(404).json({ message: 'Assessment session not found' });
    }

    // Get analytics from RL system
    const analytics = await callRLSystem('get_analytics', {
      sessionId,
      questionHistory: session.questionHistory,
      performanceHistory: session.performanceHistory
    });

    res.json({
      session: {
        track: session.track,
        status: session.status,
        startTime: session.createdAt,
        endTime: session.updatedAt,
        studentAbility: session.studentAbility,
        confidenceScore: session.confidenceScore
      },
      analytics
    });

  } catch (error) {
    console.error('Analytics error:', error);
    res.status(500).json({ message: 'Failed to get analytics', error: error.message });
  }
});

// Get adaptive recommendations based on RL analysis
app.get('/api/adaptive-recommendations/:userId', authenticateToken, async (req, res) => {
  try {
    const user = await User.findById(req.params.userId);
    
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Get RL-based recommendations
    const recommendations = await callRLSystem('get_recommendations', {
      userId: user._id,
      currentAbility: user.learningProfile.currentAbility,
      track: user.learningProfile.track,
      assessmentHistory: user.learningProfile.assessmentHistory.slice(-5), // Last 5 assessments
      masteredTopics: user.learningProfile.masteredTopics,
      strugglingTopics: user.learningProfile.strugglingTopics
    });

    res.json({
      currentAbility: user.learningProfile.currentAbility,
      recommendedTrack: user.learningProfile.track,
      recommendedDifficulty: recommendations.recommendedDifficulty,
      recommendedLessons: recommendations.lessons,
      strengths: recommendations.strengths,
      weaknesses: recommendations.weaknesses,
      learningPath: recommendations.learningPath,
      estimatedTimeToMastery: recommendations.estimatedTime
    });

  } catch (error) {
    console.error('Recommendations error:', error);
    res.status(500).json({ message: 'Failed to get recommendations', error: error.message });
  }
});

// Get user's assessment history
app.get('/api/assessment/history/:userId', authenticateToken, async (req, res) => {
  try {
    const user = await User.findById(req.params.userId);
    
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    const assessmentHistory = user.learningProfile.assessmentHistory.map(assessment => ({
      sessionId: assessment.sessionId,
      track: assessment.track,
      date: assessment.endTime || assessment.startTime,
      score: `${assessment.correctAnswers}/${assessment.totalQuestions}`,
      accuracy: assessment.totalQuestions > 0 ? (assessment.correctAnswers / assessment.totalQuestions * 100).toFixed(1) + '%' : '0%',
      finalAbility: (assessment.finalAbility * 100).toFixed(1) + '%',
      recommendedLevel: assessment.recommendedLevel
    }));

    res.json({
      assessmentHistory,
      totalAssessments: assessmentHistory.length,
      averageAbility: user.learningProfile.currentAbility,
      preferredTrack: user.learningProfile.track
    });

  } catch (error) {
    console.error('History error:', error);
    res.status(500).json({ message: 'Failed to get assessment history', error: error.message });
  }
});

// Get available tracks and their statistics
app.get('/api/tracks', async (req, res) => {
  try {
    const tracks = await callRLSystem('get_tracks', {});
    res.json(tracks);
  } catch (error) {
    console.error('Tracks error:', error);
    res.status(500).json({ message: 'Failed to get tracks', error: error.message });
  }
});

// Update learning profile based on RL insights
app.put('/api/profile/update-from-rl', authenticateToken, async (req, res) => {
  try {
    const userId = req.user.userId;
    const { ability, recommendedLevel, masteredTopics, strugglingTopics } = req.body;

    const user = await User.findById(userId);
    if (!user) {
      return res.status(404).json({ message: 'User not found' });
    }

    // Update learning profile with RL insights
    user.learningProfile.currentAbility = ability;
    user.learningProfile.preferredDifficulty = recommendedLevel;
    user.learningProfile.masteredTopics = masteredTopics;
    user.learningProfile.strugglingTopics = strugglingTopics;

    // Calculate learning velocity
    const recentAssessments = user.learningProfile.assessmentHistory.slice(-3);
    if (recentAssessments.length > 1) {
      const abilityGrowth = recentAssessments[recentAssessments.length - 1].finalAbility - 
                           recentAssessments[0].finalAbility;
      user.learningProfile.learningVelocity = abilityGrowth;
    }

    await user.save();

    res.json({
      message: 'Profile updated successfully',
      updatedProfile: user.learningProfile
    });

  } catch (error) {
    console.error('Profile update error:', error);
    res.status(500).json({ message: 'Failed to update profile', error: error.message });
  }
});

const PORT = process.env.PORT || 5000;
app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
  console.log('RL Assessment API endpoints:');
  console.log('- POST /api/assessment/start');
  console.log('- POST /api/assessment/answer');
  console.log('- GET /api/assessment/analytics/:sessionId');
  console.log('- GET /api/assessment/history/:userId');
  console.log('- GET /api/adaptive-recommendations/:userId');
  console.log('- GET /api/tracks');
});