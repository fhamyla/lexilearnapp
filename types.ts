
import { ViewProps, TextProps, TouchableOpacityProps, PressableProps, ScrollViewProps, TextInputProps, ImageProps, FlatListProps } from 'react-native';

// Removed manual augmentation causing 'module not found' error.
// Ensure NativeWind types are referenced in a separate d.ts file if needed (e.g., app.d.ts with /// <reference types="nativewind/types" />)

export enum Screen {
  LOGIN = 'LOGIN',
  SIGN_UP = 'SIGN_UP',
  PENDING_APPROVAL = 'PENDING_APPROVAL',
  TWO_FACTOR = 'TWO_FACTOR',
  ASSESSMENT = 'ASSESSMENT',
  HOME = 'HOME',
  CHILD_DASHBOARD = 'CHILD_DASHBOARD',
  LEARNING_JOURNEY = 'LEARNING_JOURNEY',
  TRACING = 'TRACING',
  READING = 'READING',
  SPELLING = 'SPELLING',
  MEMORY = 'MEMORY',
  DASHBOARD = 'DASHBOARD',
  ADMIN_DASHBOARD = 'ADMIN_DASHBOARD',
  SETTINGS = 'SETTINGS',
  DIFFICULTY_SELECT = 'DIFFICULTY_SELECT'
}

export enum Difficulty {
  MILD = 'Mild',
  MODERATE = 'Moderate',
  SEVERE = 'Severe',
  PROFOUND = 'Profound'
}

export interface AssessmentScores {
  readingSpeed: number;
  readingAccuracy: number;
  readingComprehension: number;
  writingSpeed: number;
  writingQuality: number;
  grammar: number;
  phoneticSpelling: number;
  irregularSpelling: number;
  spellingAccuracy: number;
  overallAverage: number;
  focusAreas?: string[]; // Added: Ordered list of focus areas (e.g. ['Reading', 'Writing', 'Spelling'])
}

export interface ProgressRecord {
  date: string;
  activityType: 'Tracing' | 'Reading' | 'Spelling' | 'Memory';
  score: number;
  details: string;
}

export interface UserProfile {
  uid: string;
  email: string;
  username?: string; // Generated on approval
  childName: string;
  childAge?: string;
  role: 'Guardian' | 'Admin';
  status: 'PENDING' | 'APPROVED' | 'REJECTED'; 
  assessmentComplete: boolean;
  assignedDifficulty: Difficulty;
  assessmentScores?: AssessmentScores;
  progressHistory?: ProgressRecord[];
  // Progress Tracking (Last Completed Level Index)
  lastTracingIndex?: number;
  lastReadingIndex?: number;
  lastSpellingIndex?: number;
  lastMemoryIndex?: number;
}

export interface GuardianApplication {
  id?: string;
  uid?: string; // Link to the Auth User
  guardianName: string;
  email: string;
  username?: string; // Generated on approval
  childName: string;
  childAge: string;
  relationship: string;
  difficultyRatings: {
    reading: number;
    writing: number;
    spelling: number;
  };
  status: 'PENDING' | 'APPROVED' | 'REJECTED';
  dateApplied: string;
}

export enum TracingCategory {
  LINES = 'Lines',
  LETTERS = 'Letters',
  NUMBERS = 'Numbers',
  SHAPES = 'Shapes'
}

export interface TracingVariant {
  pathData: string;
  label: string;
  viewBox?: string;
}

export interface TracingItem {
  id: string;
  category: TracingCategory;
  label: string;
  pathData: string; // Default/Fallback
  viewBox: string;
  difficultyConfig?: {
    [key in Difficulty]?: TracingVariant;
  };
}

export interface ReadingVariant {
  word: string;
  sentence: string;
}

export interface ReadingItem {
  id: string;
  difficultyLevel: number;
  word: string; // Default/Fallback
  sentence: string; // Default/Fallback
  difficultyConfig?: {
    [key in Difficulty]?: ReadingVariant;
  };
}

export interface SpellingItem {
  id: string;
  word: string;
  scrambled: string[];
  hint: string;
  contextSentence?: string; // New field for TTS context
}

export interface MemoryItem {
  id: string;
  sequence: string;
  type: 'Numbers' | 'Letters' | 'Mixed';
}

export interface PronunciationResult {
  score: number;
  isCorrect: boolean;
  feedback: string;
  transcript?: string;
  phoneticBreakdown?: string;
}

export interface ReadingAssessmentResult {
  wpm: number;
  accuracy: number;
  transcript: string;
}

export type AudioUri = string;
