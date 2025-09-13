import { Sequelize } from 'sequelize';
import dotenv from 'dotenv';
import path from 'path';
import { fileURLToPath } from 'url';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

// Use SQLite for simplicity in development
const dbPath = path.join(__dirname, '../../data/lantern_fly_tracker.db');

// Database configuration
const dbConfig = {
  dialect: 'sqlite',
  storage: dbPath,
  logging: process.env.NODE_ENV === 'development' ? console.log : false,
  pool: {
    max: 5,
    min: 0,
    acquire: 30000,
    idle: 10000,
  },
};

// Create Sequelize instance
export const sequelize = new Sequelize(dbConfig);

// Test database connection
export const testConnection = async () => {
  try {
    await sequelize.authenticate();
    console.log('✅ Database connection established successfully.');
    return true;
  } catch (error) {
    console.error('❌ Unable to connect to the database:', error);
    return false;
  }
};

// Initialize database (create tables if they don't exist)
export const initializeDatabase = async () => {
  try {
    // Import models to ensure they are registered
    await import('../models/index.js');
    
    // Sync database (create tables if they don't exist)
    await sequelize.sync({ alter: true }); // Use alter: true for development
    console.log('✅ Database synchronized successfully.');
    console.log('📊 Tables created: users, photos');
    return true;
  } catch (error) {
    console.error('❌ Database synchronization failed:', error);
    return false;
  }
};
