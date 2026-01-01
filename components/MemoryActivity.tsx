
import React, { useState, useEffect } from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { MemoryItem, Difficulty } from '../types';
import { ChevronLeft, Delete, Eye, CheckCircle, AlertCircle, Play } from 'lucide-react-native';

interface MemoryActivityProps {
  items: MemoryItem[];
  difficulty: Difficulty;
  initialIndex?: number;
  onComplete: (score: number) => void;
  onLevelComplete?: (levelIndex: number) => void;
  onExit: () => void;
}

export const MemoryActivity: React.FC<MemoryActivityProps> = ({ items, difficulty, onComplete, onLevelComplete, onExit, initialIndex = 0 }) => {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [phase, setPhase] = useState<'IDLE' | 'SHOW' | 'INPUT' | 'RESULT'>('IDLE');
  const [inputValue, setInputValue] = useState('');
  const [timeLeft, setTimeLeft] = useState(0);
  const [resultMessage, setResultMessage] = useState<'Correct' | 'Incorrect' | null>(null);

  const currentItem = items[currentIndex];
  
  const getDisplayTime = () => {
    switch(difficulty) {
      case Difficulty.MILD: return 3000;
      case Difficulty.MODERATE: return 2000;
      case Difficulty.SEVERE: return 1500;
      case Difficulty.PROFOUND: return 1000;
      default: return 3000;
    }
  };

  const initialTime = getDisplayTime();

  useEffect(() => {
    let timer: NodeJS.Timeout;
    if (phase === 'SHOW' && timeLeft > 0) {
      timer = setInterval(() => {
        setTimeLeft((prev) => {
          if (prev <= 100) {
            setPhase('INPUT');
            return 0;
          }
          return prev - 100;
        });
      }, 100);
    }
    return () => clearInterval(timer);
  }, [phase, timeLeft]);

  const startLevel = () => {
    setInputValue('');
    setResultMessage(null);
    setTimeLeft(initialTime);
    setPhase('SHOW');
  };

  const handleKeypadPress = (key: string) => {
    if (key === 'DEL') {
        setInputValue(prev => prev.slice(0, -1));
    } else if (key === 'SUBMIT') {
        handleSubmit();
    } else {
        setInputValue(prev => prev + key);
    }
  };

  const handleSubmit = () => {
    setPhase('RESULT');
    const normalizedInput = inputValue.replace(/\s/g, '').toUpperCase();
    const normalizedTarget = currentItem.sequence.replace(/\s/g, '').toUpperCase();
    
    if (normalizedInput === normalizedTarget) {
      setResultMessage('Correct');
    } else {
      setResultMessage('Incorrect');
    }
  };

  const nextLevel = () => {
    if (resultMessage === 'Correct') {
        onComplete(100); 
        if (onLevelComplete) {
            onLevelComplete(currentIndex); // Save current level (0-based) as completed
        }
    }
    
    if (currentIndex < items.length - 1) {
      setCurrentIndex(prev => prev + 1);
      setPhase('IDLE');
      setInputValue('');
      setResultMessage(null);
    } else {
        onComplete(100);
        onExit();
    }
  };

  const retryLevel = () => {
      setPhase('IDLE');
      setInputValue('');
      setResultMessage(null);
  };

  const renderNumpad = () => (
      <View style={styles.numpadContainer}>
          {[1,2,3,4,5,6,7,8,9].map(n => (
              <TouchableOpacity key={n} onPress={() => handleKeypadPress(n.toString())} style={styles.numpadButton}>
                  <Text style={styles.numpadText}>{n}</Text>
              </TouchableOpacity>
          ))}
          <TouchableOpacity onPress={() => setInputValue('')} style={[styles.numpadButton, styles.numpadClear]}>
             <Text style={styles.numpadClearText}>Clear</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => handleKeypadPress('0')} style={styles.numpadButton}>
             <Text style={styles.numpadText}>0</Text>
          </TouchableOpacity>
          <TouchableOpacity onPress={() => handleKeypadPress('DEL')} style={[styles.numpadButton, styles.numpadDel]}>
             <Delete size={28} stroke="#374151" />
          </TouchableOpacity>
          
          <TouchableOpacity onPress={() => handleKeypadPress('SUBMIT')} style={styles.submitButton}>
             <Text style={styles.submitButtonText}>Submit Answer</Text>
          </TouchableOpacity>
      </View>
  );

  const renderKeyboard = () => {
      const rows = [
          ['Q','W','E','R','T','Y','U','I','O','P'],
          ['A','S','D','F','G','H','J','K','L'],
          ['Z','X','C','V','B','N','M']
      ];
      return (
          <View style={styles.keyboardContainer}>
              {rows.map((row, i) => (
                  <View key={i} style={styles.keyboardRow}>
                      {row.map(char => (
                          <TouchableOpacity key={char} onPress={() => handleKeypadPress(char)} style={styles.keyboardKey}>
                             <Text style={styles.keyboardKeyText}>{char}</Text>
                          </TouchableOpacity>
                      ))}
                  </View>
              ))}
              
              {/* Number Row for Mixed */}
              <View style={[styles.keyboardRow, {marginTop: 8}]}>
                 {[1,2,3,4,5,6,7,8,9,0].map(n => (
                    <TouchableOpacity key={n} onPress={() => handleKeypadPress(n.toString())} style={styles.numberKey}>
                        <Text style={styles.numberKeyText}>{n}</Text>
                    </TouchableOpacity>
                 ))}
              </View>

              <View style={styles.keyboardActions}>
                   <TouchableOpacity onPress={() => handleKeypadPress('DEL')} style={styles.keyDelButton}>
                      <Delete size={20} stroke="#DC2626" />
                      <Text style={styles.keyDelText}>DEL</Text>
                   </TouchableOpacity>
                   <TouchableOpacity onPress={() => handleKeypadPress('SUBMIT')} style={styles.keySubmitButton}>
                      <Text style={styles.submitButtonText}>Submit</Text>
                   </TouchableOpacity>
              </View>
          </View>
      );
  };

  return (
    <View style={styles.container}>
       {/* Header */}
       <View style={styles.header}>
        <TouchableOpacity onPress={onExit} style={styles.backButton}>
          <ChevronLeft size={24} stroke="#4B5563" />
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>
        <View style={styles.titleContainer}>
            <Text style={styles.difficultyLabel}>{difficulty} Difficulty</Text>
            <Text style={styles.levelLabel}>Level {currentIndex + 1}/40</Text>
        </View>
        <View style={{width: 60}}></View>
      </View>

      <View style={styles.card}>
        
        {/* Progress Bar for Timer */}
        {phase === 'SHOW' && (
            <View style={styles.timerBg}>
                <View 
                    style={[styles.timerFill, { width: `${(timeLeft / initialTime) * 100}%` }]} 
                />
            </View>
        )}

        {/* Content Area */}
        <View style={styles.contentArea}>
            
            {phase === 'IDLE' && (
                <View style={styles.centerCol}>
                    <View style={styles.iconCircle}>
                        <Eye size={48} stroke="#4A90E2" />
                    </View>
                    <Text style={styles.readyTitle}>Ready?</Text>
                    <Text style={styles.readyDesc}>
                        Memorize the sequence shown on screen. It will disappear quickly!
                    </Text>
                    <TouchableOpacity onPress={startLevel} style={styles.startButton}>
                        <Play size={24} stroke="#FFF" {...({fill: "#FFF"} as any)} />
                        <Text style={styles.startButtonText}>Start Level</Text>
                    </TouchableOpacity>
                </View>
            )}

            {phase === 'SHOW' && (
                <View style={styles.centerCol}>
                    <Text style={styles.promptLabel}>Memorize This</Text>
                    <View style={styles.sequenceBox}>
                        <Text style={styles.sequenceText}>{currentItem.sequence}</Text>
                    </View>
                </View>
            )}

            {phase === 'INPUT' && (
                <View style={styles.centerCol}>
                    <Text style={styles.promptLabel}>What was it?</Text>
                    
                    {/* Input Display */}
                    <View style={styles.inputDisplay}>
                        <Text style={styles.inputText}>
                            {inputValue || <Text style={{color: '#D1D5DB'}}>_ _ _</Text>}
                        </Text>
                    </View>

                    {currentItem.type === 'Numbers' ? renderNumpad() : renderKeyboard()}
                </View>
            )}

            {phase === 'RESULT' && (
                <View style={styles.centerCol}>
                     {resultMessage === 'Correct' ? (
                         <View style={styles.centerCol}>
                            <CheckCircle size={80} stroke="#16A34A" style={{marginBottom: 16}} />
                            <Text style={styles.correctTitle}>Correct!</Text>
                            <Text style={styles.resultDesc}>You have a great memory.</Text>
                            <TouchableOpacity onPress={nextLevel} style={styles.nextButton}>
                                <Text style={styles.nextButtonText}>Next Level</Text>
                            </TouchableOpacity>
                         </View>
                     ) : (
                        <View style={styles.centerCol}>
                            <AlertCircle size={80} stroke="#EF4444" style={{marginBottom: 16}} />
                            <Text style={styles.incorrectTitle}>Incorrect</Text>
                            <Text style={styles.resultDesc}>The sequence was:</Text>
                            <Text style={styles.correctSequence}>{currentItem.sequence}</Text>
                            <TouchableOpacity onPress={retryLevel} style={styles.retryButton}>
                                <Text style={styles.retryButtonText}>Try Again</Text>
                            </TouchableOpacity>
                        </View>
                     )}
                </View>
            )}
        </View>

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
  difficultyLabel: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#9CA3AF',
    textTransform: 'uppercase',
    letterSpacing: 1,
  },
  levelLabel: {
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
    borderColor: '#E6F3F7',
    overflow: 'hidden',
  },
  timerBg: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    height: 8,
    backgroundColor: '#F3F4F6',
  },
  timerFill: {
    height: '100%',
    backgroundColor: '#4A90E2',
  },
  contentArea: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
    width: '100%',
  },
  centerCol: {
    alignItems: 'center',
    width: '100%',
  },
  iconCircle: {
    width: 96,
    height: 96,
    backgroundColor: '#DBEAFE',
    borderRadius: 999,
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 24,
  },
  readyTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 8,
  },
  readyDesc: {
    color: '#6B7280',
    textAlign: 'center',
    marginBottom: 32,
    paddingHorizontal: 32,
  },
  startButton: {
    paddingHorizontal: 48,
    paddingVertical: 16,
    backgroundColor: '#4A90E2',
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
  startButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 20,
  },
  promptLabel: {
    color: '#9CA3AF',
    fontWeight: 'bold',
    marginBottom: 16,
    textTransform: 'uppercase',
    letterSpacing: 2,
  },
  sequenceBox: {
    backgroundColor: '#F9FAFB',
    paddingHorizontal: 48,
    paddingVertical: 32,
    borderRadius: 16,
    borderWidth: 2,
    borderColor: '#D1D5DB',
    borderStyle: 'dashed',
  },
  sequenceText: {
    fontSize: 60,
    fontWeight: 'bold',
    color: '#4A90E2',
    letterSpacing: 8,
  },
  inputDisplay: {
    width: '100%',
    height: 80,
    backgroundColor: '#F9FAFB',
    borderRadius: 12,
    borderWidth: 2,
    borderColor: 'rgba(74, 144, 226, 0.3)',
    alignItems: 'center',
    justifyContent: 'center',
    marginBottom: 16,
  },
  inputText: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#1F2937',
    letterSpacing: 4,
  },
  numpadContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    justifyContent: 'center',
    gap: 16,
    maxWidth: 320,
    marginTop: 24,
  },
  numpadButton: {
    width: 80,
    height: 80,
    backgroundColor: '#FFFFFF',
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
    alignItems: 'center',
    justifyContent: 'center',
    borderWidth: 1,
    borderColor: '#F3F4F6',
  },
  numpadText: {
    fontSize: 30,
    fontWeight: 'bold',
    color: '#374151',
  },
  numpadClear: {
    backgroundColor: '#FEF2F2',
    borderColor: '#FEE2E2',
  },
  numpadClearText: {
    color: '#DC2626',
    fontWeight: 'bold',
    fontSize: 18,
  },
  numpadDel: {
    backgroundColor: '#F3F4F6',
    borderColor: '#E5E7EB',
  },
  submitButton: {
    width: '100%',
    paddingVertical: 16,
    marginTop: 8,
    backgroundColor: '#4A90E2',
    borderRadius: 12,
    alignItems: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  submitButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 20,
  },
  keyboardContainer: {
    width: '100%',
    alignItems: 'center',
    gap: 8,
    marginTop: 16,
  },
  keyboardRow: {
    flexDirection: 'row',
    gap: 4,
  },
  keyboardKey: {
    width: 32,
    height: 48,
    backgroundColor: '#FFFFFF',
    borderRadius: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 1,
    elevation: 1,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    alignItems: 'center',
    justifyContent: 'center',
  },
  keyboardKeyText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#374151',
  },
  numberKey: {
    width: 32,
    height: 40,
    backgroundColor: '#F9FAFB',
    borderRadius: 4,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 1,
    elevation: 1,
    borderWidth: 1,
    borderColor: '#E5E7EB',
    alignItems: 'center',
    justifyContent: 'center',
  },
  numberKeyText: {
    fontWeight: 'bold',
    color: '#4B5563',
  },
  keyboardActions: {
    flexDirection: 'row',
    gap: 16,
    marginTop: 16,
    width: '100%',
    paddingHorizontal: 16,
  },
  keyDelButton: {
    paddingHorizontal: 24,
    paddingVertical: 16,
    backgroundColor: '#FEE2E2',
    borderRadius: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.05,
    shadowRadius: 2,
    elevation: 1,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  keyDelText: {
    color: '#DC2626',
    fontWeight: 'bold',
  },
  keySubmitButton: {
    flex: 1,
    backgroundColor: '#4A90E2',
    borderRadius: 12,
    alignItems: 'center',
    justifyContent: 'center',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 2,
  },
  correctTitle: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#16A34A',
    marginBottom: 8,
  },
  incorrectTitle: {
    fontSize: 36,
    fontWeight: 'bold',
    color: '#EF4444',
    marginBottom: 8,
  },
  resultDesc: {
    color: '#6B7280',
    marginBottom: 32,
  },
  correctSequence: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 32,
  },
  nextButton: {
    paddingHorizontal: 48,
    paddingVertical: 16,
    backgroundColor: '#22C55E',
    borderRadius: 999,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  nextButtonText: {
    color: '#FFFFFF',
    fontWeight: 'bold',
    fontSize: 20,
  },
  retryButton: {
    paddingHorizontal: 48,
    paddingVertical: 16,
    backgroundColor: '#E5E7EB',
    borderRadius: 999,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  retryButtonText: {
    color: '#374151',
    fontWeight: 'bold',
    fontSize: 20,
  },
});
