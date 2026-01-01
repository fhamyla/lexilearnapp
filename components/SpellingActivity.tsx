import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { SpellingItem, Difficulty } from '../types';
import { ChevronLeft, ChevronRight, Volume2, RotateCcw, CheckCircle } from 'lucide-react-native';
import * as Speech from 'expo-speech';

interface SpellingActivityProps {
  items: SpellingItem[];
  difficulty: Difficulty;
  initialIndex?: number;
  onComplete: (score: number) => void;
  onLevelComplete?: (levelIndex: number) => void;
  onExit: () => void;
}

export const SpellingActivity: React.FC<SpellingActivityProps> = ({ items, difficulty, onComplete, onLevelComplete, onExit, initialIndex = 0 }) => {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [unlockedLevel, setUnlockedLevel] = useState(initialIndex);
  const [currentLetters, setCurrentLetters] = useState<string[]>([]);
  const [userAnswer, setUserAnswer] = useState<(string | null)[]>([]);
  const [isCorrect, setIsCorrect] = useState(false);
  const [feedback, setFeedback] = useState<string>('');

  const currentItem = items[currentIndex];

  useEffect(() => {
    resetLevel();
  }, [currentIndex, currentItem]);

  const resetLevel = () => {
    const letters = currentItem.word.split('');
    const shuffled = [...letters].sort(() => Math.random() - 0.5);
    setCurrentLetters(shuffled);
    setUserAnswer(new Array(letters.length).fill(null));
    setIsCorrect(false);
    setFeedback('');
  };

  const playWord = () => {
    Speech.stop();
    const textToSpeak = currentItem.contextSentence 
        ? `${currentItem.word}. ${currentItem.contextSentence}`
        : currentItem.word;
    Speech.speak(textToSpeak, { rate: 0.9 });
  };

  const playHint = () => {
    Speech.stop();
    Speech.speak(currentItem.hint);
  };

  const playLetter = (letter: string) => {
    Speech.stop();
    Speech.speak(letter, { rate: 0.9 });
  };

  const handleLetterClick = (letter: string, index: number) => {
    if (isCorrect) return;

    playLetter(letter);

    const firstEmptyIndex = userAnswer.findIndex(val => val === null);
    if (firstEmptyIndex !== -1) {
      const newAnswer = [...userAnswer];
      newAnswer[firstEmptyIndex] = letter;
      setUserAnswer(newAnswer);

      const newLetters = [...currentLetters];
      newLetters.splice(index, 1);
      setCurrentLetters(newLetters);

      if (firstEmptyIndex === userAnswer.length - 1) {
        checkAnswer(newAnswer as string[]);
      }
    }
  };

  const handleSlotClick = (index: number) => {
    if (isCorrect) return;
    const letter = userAnswer[index];
    if (letter) {
      playLetter(letter);
      setCurrentLetters([...currentLetters, letter]);
      const newAnswer = [...userAnswer];
      newAnswer[index] = null;
      setUserAnswer(newAnswer);
      setFeedback('');
    }
  };

  const checkAnswer = (answer: string[]) => {
    const joined = answer.join('');
    if (joined === currentItem.word) {
      setIsCorrect(true);
      setFeedback('Correct!');
      Speech.speak(`Correct! The word is ${currentItem.word}`);
      const score = 100;
      onComplete(score);
      
      // UNLOCK NEXT IF CORRECT
      if (currentIndex >= unlockedLevel) {
          const nextLevel = Math.max(unlockedLevel, currentIndex + 1);
          setUnlockedLevel(nextLevel);
          if (onLevelComplete) {
              onLevelComplete(nextLevel - 1);
          }
      }
    } else {
      setFeedback('Try again!');
      Speech.speak("Try again");
    }
  };

  const nextLevel = () => {
    if (currentIndex < items.length - 1) {
      setCurrentIndex(prev => prev + 1);
    }
  };

  const prevLevel = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
    }
  };

  return (
    <View style={styles.container}>
      {/* Header */}
      <View style={styles.header}>
        <TouchableOpacity onPress={onExit} style={styles.backButton}>
          <ChevronLeft size={24} {...({color: "#4B5563"} as any)} /> 
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>
        <View style={styles.titleContainer}>
          <Text style={styles.difficultyText}>{difficulty} Difficulty</Text>
          <Text style={styles.levelText}>Level {currentIndex + 1}/40</Text>
        </View>
        <View style={{width: 60}}></View>
      </View>

      <View style={styles.card}>
        
        {/* Navigation */}
        <TouchableOpacity onPress={prevLevel} disabled={currentIndex === 0} style={[styles.navButton, styles.navLeft, currentIndex === 0 && styles.disabledNav]}>
          <ChevronLeft size={24} {...({color: "#000"} as any)} />
        </TouchableOpacity>
        
        <TouchableOpacity 
            onPress={nextLevel} 
            disabled={currentIndex === items.length - 1 || currentIndex >= unlockedLevel} 
            style={[
                styles.navButton, 
                styles.navRight, 
                (currentIndex === items.length - 1 || currentIndex >= unlockedLevel) && styles.disabledNav
            ]}
        >
          <ChevronRight size={24} {...({color: currentIndex >= unlockedLevel ? "#9CA3AF" : "#000"} as any)} />
        </TouchableOpacity>

        {/* Word / Image Area */}
        <View style={styles.wordArea}>
          <View style={styles.controlsRow}>
            <TouchableOpacity 
                onPress={playWord} 
                style={styles.hearButton}
            >
              <Volume2 size={24} {...({color: "#FFF"} as any)} /> 
              <Text style={styles.hearButtonText}>Hear Word</Text>
            </TouchableOpacity>
            <TouchableOpacity onPress={playHint} style={styles.hintButton}>
              <Text style={styles.hintText}>Hint: {currentItem.hint}</Text>
            </TouchableOpacity>
          </View>
          <Text style={styles.helperText}>Click the speaker to hear context.</Text>
        </View>

        {/* Slots */}
        <View style={styles.slotsContainer}>
          {userAnswer.map((char, idx) => (
            <TouchableOpacity
              key={idx}
              onPress={() => handleSlotClick(idx)}
              style={[
                  styles.slot,
                  char ? styles.slotFilled : styles.slotEmpty,
                  isCorrect && styles.slotCorrect
              ]}
            >
              {char ? (
                  <Text style={[styles.slotText, isCorrect ? styles.textCorrect : styles.textFilled]}>{char}</Text>
              ) : (
                  difficulty === Difficulty.PROFOUND ? <Text style={styles.slotPlaceholder}>{currentItem.word[idx]}</Text> : null
              )}
            </TouchableOpacity>
          ))}
        </View>

        {/* Feedback */}
        {feedback ? (
          <View style={styles.feedbackContainer}>
            {isCorrect && <CheckCircle size={24} {...({color: "#16A34A"} as any)} />}
            <Text style={[styles.feedbackText, isCorrect ? styles.textSuccess : styles.textError]}>{feedback}</Text>
          </View>
        ) : null}

        {/* Letter Bank */}
        <View style={styles.letterBank}>
          {currentLetters.map((letter, idx) => (
            <TouchableOpacity
              key={`${letter}-${idx}`}
              onPress={() => handleLetterClick(letter, idx)}
              disabled={isCorrect}
              style={styles.letterButton}
            >
              <Text style={styles.letterText}>{letter}</Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* Reset */}
        <TouchableOpacity onPress={resetLevel} style={styles.resetButton}>
          <RotateCcw size={16} {...({color: "#9CA3AF"} as any)} />
          <Text style={styles.resetText}>Reset Level</Text>
        </TouchableOpacity>

      </View>
    </View>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FDFBF7',
    padding: 16,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 24,
  },
  backButton: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  backText: {
    color: '#4B5563',
    fontWeight: 'bold',
  },
  titleContainer: {
    alignItems: 'center',
  },
  difficultyText: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#9CA3AF',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  levelText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#2D2D2D',
  },
  card: {
    backgroundColor: '#FFFFFF',
    borderRadius: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 10,
    elevation: 4,
    padding: 24,
    flex: 1,
    alignItems: 'center',
    position: 'relative',
    borderWidth: 4,
    borderColor: '#FEF08A',
  },
  navButton: {
    position: 'absolute',
    top: '50%',
    marginTop: -20,
    padding: 12,
    borderRadius: 999,
    backgroundColor: '#F9FAFB',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
    zIndex: 10,
  },
  navLeft: {
    left: 16,
  },
  navRight: {
    right: 16,
  },
  disabledNav: {
    opacity: 0.3,
    backgroundColor: '#F3F4F6',
  },
  wordArea: {
    alignItems: 'center',
    marginTop: 16,
    marginBottom: 32,
    width: '100%',
  },
  controlsRow: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 16,
    justifyContent: 'center',
    flexWrap: 'wrap',
  },
  hearButton: {
    paddingHorizontal: 24,
    paddingVertical: 16,
    backgroundColor: '#3B82F6',
    borderRadius: 999,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  hearButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 18,
  },
  hintButton: {
    paddingHorizontal: 16,
    paddingVertical: 8,
    backgroundColor: '#FEF9C3',
    borderRadius: 999,
    justifyContent: 'center',
  },
  hintText: {
    color: '#A16207',
    fontWeight: '600',
    fontSize: 12,
  },
  helperText: {
    fontSize: 14,
    color: '#9CA3AF',
    textAlign: 'center',
  },
  slotsContainer: {
    flexDirection: 'row',
    gap: 8,
    marginBottom: 48,
    flexWrap: 'wrap',
    justifyContent: 'center',
  },
  slot: {
    width: 56,
    height: 56,
    borderBottomWidth: 4,
    alignItems: 'center',
    justifyContent: 'center',
    borderRadius: 8,
  },
  slotFilled: {
    backgroundColor: '#EFF6FF',
    borderColor: '#60A5FA',
  },
  slotEmpty: {
    backgroundColor: '#F3F4F6',
    borderColor: '#D1D5DB',
  },
  slotCorrect: {
    backgroundColor: '#DCFCE7',
    borderColor: '#22C55E',
  },
  slotText: {
    fontSize: 30,
    fontWeight: 'bold',
  },
  textFilled: {
    color: '#1E40AF',
  },
  textCorrect: {
    color: '#15803D',
  },
  slotPlaceholder: {
    fontSize: 30,
    fontWeight: 'bold',
    color: '#E5E7EB',
  },
  feedbackContainer: {
    marginBottom: 24,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  feedbackText: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  textSuccess: {
    color: '#16A34A',
  },
  textError: {
    color: '#F97316',
  },
  letterBank: {
    flexDirection: 'row',
    gap: 12,
    flexWrap: 'wrap',
    justifyContent: 'center',
    maxWidth: 384,
    backgroundColor: '#F9FAFB',
    padding: 24,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#E5E7EB',
  },
  letterButton: {
    width: 48,
    height: 48,
    backgroundColor: '#FFFFFF',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#E5E7EB',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
  },
  letterText: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#374151',
  },
  resetButton: {
    marginTop: 'auto',
    marginBottom: 8,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  resetText: {
    color: '#9CA3AF',
    fontSize: 14,
  },
});