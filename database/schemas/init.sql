-- Lantern Fly Tracker Database Schema
-- This file contains the initial database schema

-- Enable PostGIS extension for geospatial data
CREATE EXTENSION IF NOT EXISTS postgis;

-- Users table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    email VARCHAR(100) UNIQUE,
    points INTEGER DEFAULT 0,
    total_photos INTEGER DEFAULT 0,
    confirmed_lantern_flies INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Photos table
CREATE TABLE photos (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    image_url VARCHAR(500) NOT NULL,
    thumbnail_url VARCHAR(500),
    latitude DECIMAL(10, 8),
    longitude DECIMAL(11, 8),
    location_name VARCHAR(200),
    is_lantern_fly BOOLEAN,
    confidence_score DECIMAL(5, 4),
    points_awarded INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create spatial index for location queries
CREATE INDEX idx_photos_location ON photos USING GIST (ST_Point(longitude, latitude));

-- Photo embeddings table (for future duplication detection)
CREATE TABLE photo_embeddings (
    id SERIAL PRIMARY KEY,
    photo_id INTEGER REFERENCES photos(id) ON DELETE CASCADE,
    embedding VECTOR(512), -- For similarity comparison
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT NOW()
);

-- User achievements table
CREATE TABLE achievements (
    id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
    achievement_type VARCHAR(50) NOT NULL,
    achievement_name VARCHAR(100) NOT NULL,
    description TEXT,
    points_reward INTEGER DEFAULT 0,
    earned_at TIMESTAMP DEFAULT NOW()
);

-- Leaderboard view
CREATE VIEW leaderboard AS
SELECT 
    u.id,
    u.username,
    u.points,
    u.total_photos,
    u.confirmed_lantern_flies,
    ROUND(
        CASE 
            WHEN u.total_photos > 0 
            THEN (u.confirmed_lantern_flies::DECIMAL / u.total_photos) * 100 
            ELSE 0 
        END, 2
    ) as accuracy_percentage
FROM users u
ORDER BY u.points DESC, u.confirmed_lantern_flies DESC;

-- Function to update user stats when photo is added
CREATE OR REPLACE FUNCTION update_user_stats()
RETURNS TRIGGER AS $$
BEGIN
    UPDATE users 
    SET 
        total_photos = total_photos + 1,
        confirmed_lantern_flies = CASE 
            WHEN NEW.is_lantern_fly = true 
            THEN confirmed_lantern_flies + 1 
            ELSE confirmed_lantern_flies 
        END,
        points = points + COALESCE(NEW.points_awarded, 0),
        updated_at = NOW()
    WHERE id = NEW.user_id;
    
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update user stats
CREATE TRIGGER trigger_update_user_stats
    AFTER INSERT ON photos
    FOR EACH ROW
    EXECUTE FUNCTION update_user_stats();

-- Function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Triggers for updated_at timestamps
CREATE TRIGGER trigger_users_updated_at
    BEFORE UPDATE ON users
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER trigger_photos_updated_at
    BEFORE UPDATE ON photos
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
