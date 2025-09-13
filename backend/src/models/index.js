import { User } from './User.js';
import { Photo } from './Photo.js';

// Define associations
User.associate({ User, Photo });
Photo.associate({ User, Photo });

export { User, Photo };
