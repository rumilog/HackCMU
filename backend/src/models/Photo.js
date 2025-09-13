import { DataTypes } from 'sequelize';
import { sequelize } from '../config/database.js';

export const Photo = sequelize.define('Photo', {
  id: {
    type: DataTypes.INTEGER,
    primaryKey: true,
    autoIncrement: true,
  },
  user_id: {
    type: DataTypes.INTEGER,
    allowNull: false,
    references: {
      model: 'users',
      key: 'id',
    },
  },
  image_url: {
    type: DataTypes.STRING(500),
    allowNull: false,
  },
  thumbnail_url: {
    type: DataTypes.STRING(500),
    allowNull: true,
  },
  latitude: {
    type: DataTypes.DECIMAL(10, 8),
    allowNull: true,
    validate: {
      min: -90,
      max: 90,
    },
  },
  longitude: {
    type: DataTypes.DECIMAL(11, 8),
    allowNull: true,
    validate: {
      min: -180,
      max: 180,
    },
  },
  location_name: {
    type: DataTypes.STRING(200),
    allowNull: true,
  },
  is_lantern_fly: {
    type: DataTypes.BOOLEAN,
    allowNull: true,
  },
  confidence_score: {
    type: DataTypes.DECIMAL(5, 4),
    allowNull: true,
    validate: {
      min: 0,
      max: 1,
    },
  },
  points_awarded: {
    type: DataTypes.INTEGER,
    defaultValue: 0,
    validate: {
      min: 0,
    },
  },
  created_at: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW,
  },
  updated_at: {
    type: DataTypes.DATE,
    defaultValue: DataTypes.NOW,
  },
}, {
  tableName: 'photos',
  timestamps: true,
  createdAt: 'created_at',
  updatedAt: 'updated_at',
});

// Define associations
Photo.associate = (models) => {
  // A photo belongs to a user
  Photo.belongsTo(models.User, {
    foreignKey: 'user_id',
    as: 'user',
  });
};
