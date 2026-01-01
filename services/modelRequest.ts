import { PronunciationResult } from '../types';
import { Platform } from 'react-native';
import { File } from 'expo-file-system';
import * as FileSystem from 'expo-file-system/legacy';

// Utility to log APK vs Expo environment
const isApk = Platform.OS === 'android' && !__DEV__;
const log = (...args: any[]) => {
  if (__DEV__ || isApk) {
    console.log(...args);
  }
};

// Convert URI to base64 using the new File API
const uriToBase64 = async (uri: string): Promise<string> => {
  try {
    const base64 = await FileSystem.readAsStringAsync(uri, { encoding: 'base64' });
    return base64;
  } catch (error) {
    console.log('Error converting audio to base64:', error);
    throw error;
  }
};

// Wrapper for fetch with logging
const fetchWithLogging = async (url: string, options: any) => {
  try {
    log('📡 Sending request to:', url);
    const res = await fetch(url, options);
    log('📥 Response status:', res.status);
    const text = await res.text();
    log('📄 Response text:', text.substring(0, 300)); // log first 300 chars
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    return JSON.parse(text);
  } catch (error) {
    log('❌ Network/Backend error:', error);
    throw error;
  }
};

export const checkPronunciation = async (
  audioUri: string,
  targetWord: string
): Promise<PronunciationResult> => {
  try {
    const base64Audio = await uriToBase64(audioUri);
    return await fetchWithLogging(
      'https://sesefi-lexi-reading-pronunciation-api.hf.space/pronunciation',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_base64: base64Audio, target_word: targetWord }),
      }
    );
  } catch (error) {
    return {
      score: 0,
      isCorrect: false,
      feedback: `Network error: ${(error as Error).message}`,
      transcript: '',
      phoneticBreakdown: '',
    };
  }
};

export const analyzeReadingAssessment = async (
  audioUri: string,
  targetText: string,
  durationSeconds: number
): Promise<{ wpm: number; accuracy: number; transcript: string }> => {
  try {
    const base64Audio = await uriToBase64(audioUri);
    const result = await fetchWithLogging(
      'https://sesefi-lexi-reading-pronunciation-api.hf.space/reading',
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ audio_base64: base64Audio, target_text: targetText }),
      }
    );
    const minutes = durationSeconds / 60;
    const wpm = minutes > 0 ? Math.round((result.correctWordCount || 0) / minutes) : 0;
    return { wpm, accuracy: result.accuracy || 0, transcript: result.transcript || '' };
  } catch (error) {
    return { wpm: 0, accuracy: 0, transcript: '' };
  }
};
