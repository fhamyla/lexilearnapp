import React, { useState, useEffect, useRef } from 'react';
import { View, Text, TouchableOpacity, ActivityIndicator, StyleSheet, Platform, Alert, ScrollView } from 'react-native';
import { ReadingItem, Difficulty } from '../types';
import { Mic, Volume2, Square, ArrowRight, ChevronLeft, ChevronRight, MessageSquare, Star, ThumbsUp, AlertCircle } from 'lucide-react-native';
import { checkPronunciation } from '../services/modelRequest';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';

interface ReadingActivityProps {
  items: ReadingItem[];
  difficulty: Difficulty;
  initialIndex?: number;
  onComplete: (score: number) => void;
  onLevelComplete?: (levelIndex: number) => void;
  onExit: () => void;
}

export const ReadingActivity: React.FC<ReadingActivityProps> = ({ items, onComplete, onLevelComplete, onExit, difficulty, initialIndex = 0 }) => {
  const [currentIndex, setCurrentIndex] = useState(initialIndex);
  const [unlockedLevel, setUnlockedLevel] = useState(initialIndex); 
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [feedback, setFeedback] = useState<{score: number, text: string, breakdown?: string, transcript?: string} | null>(null);

  const currentItem = items[currentIndex];
  const difficultyConfig = currentItem.difficultyConfig?.[difficulty];
  const activeWord = difficultyConfig?.word || currentItem.word;
  const activeSentence = difficultyConfig?.sentence || currentItem.sentence;

  const safeCleanupRecording = async (rec: Audio.Recording | null) => {
      if (!rec) return;
      try {
          await rec.stopAndUnloadAsync();
      } catch (err) {
      }
  };

  useEffect(() => {
    return () => {
      if (recordingRef.current) {
        safeCleanupRecording(recordingRef.current);
        recordingRef.current = null;
      }
    };
  }, []);

  const playWord = () => {
    Speech.speak(activeWord, { rate: 0.8 });
  };

  const playSentence = () => {
    Speech.speak(activeSentence, { rate: 0.9 });
  };

  const startRecording = async () => {
    setFeedback(null);
    try {
      if (recordingRef.current) {
          await safeCleanupRecording(recordingRef.current);
          recordingRef.current = null;
          setRecording(null);
      }

      const permission = await Audio.requestPermissionsAsync();
      if (permission.status !== 'granted') {
        Alert.alert("Permission", "Microphone access is required.");
        return;
      }

      await Audio.setAudioModeAsync({
        allowsRecordingIOS: true,
        playsInSilentModeIOS: true,
        staysActiveInBackground: false,
        shouldDuckAndroid: true,
        playThroughEarpieceAndroid: false,
      });

      const newRecording = new Audio.Recording();
      await newRecording.prepareToRecordAsync(Audio.RecordingOptionsPresets.HIGH_QUALITY);
      await newRecording.startAsync();
      
      recordingRef.current = newRecording;
      setRecording(newRecording);
      console.log("Recording started");

    } catch (err) {
      console.error('Failed to start recording', err);
      Alert.alert("Error", "Could not start recording.");
    }
  };

  const stopRecording = async () => {
    if (!recordingRef.current || isProcessing) return;

    setIsProcessing(true);
    let uri: string | null = null;
    
    try {
      const status = await recordingRef.current.getStatusAsync();
      
      if (status.durationMillis < 1000) {
          Alert.alert("Too Short", "Please press and speak for at least a second.");
          setIsProcessing(false);
          return;
      }

      await recordingRef.current.stopAndUnloadAsync();
      uri = recordingRef.current.getURI();
    } catch (e: any) {
      if (e.message && !e.message.includes("no valid audio data")) {
          console.log('Error stopping recording:', e);
      } else {
          Alert.alert("Audio Error", "No audio detected. Please try again.");
      }
      
      setIsProcessing(false);
      recordingRef.current = null;
      setRecording(null);
      return;
    }
    
    recordingRef.current = null;
    setRecording(null);

    if (uri) {
        console.log("Audio URI:", uri);
        const result = await checkPronunciation(uri, activeWord);
        setFeedback({
            score: result.score,
            text: result.feedback,
            breakdown: result.phoneticBreakdown,
            transcript: result.transcript
        });
        onComplete(result.score);
        
        if (result.score > 60 && currentIndex >= unlockedLevel) {
            const nextLevel = Math.max(unlockedLevel, currentIndex + 1);
            setUnlockedLevel(nextLevel);
            if (onLevelComplete) {
                onLevelComplete(nextLevel - 1);
            }
        }
    }
    setIsProcessing(false);
  };

  const nextWord = () => {
    if (currentIndex < items.length - 1) {
      setCurrentIndex(prev => prev + 1);
      setFeedback(null);
    }
  };

  const prevWord = () => {
    if (currentIndex > 0) {
      setCurrentIndex(prev => prev - 1);
      setFeedback(null);
    }
  };

  const getFeedbackStyles = (score: number) => {
      if (score >= 90) return { bg: '#DCFCE7', border: '#86EFAC', text: '#166534', icon: <Star {...({color: "#16A34A"} as any)} {...({fill: "#16A34A"} as any)} size={32} /> };
      if (score >= 70) return { bg: '#DBEAFE', border: '#93C5FD', text: '#1E40AF', icon: <ThumbsUp {...({color: "#2563EB"} as any)} size={32} /> };
      if (score >= 40) return { bg: '#FFEDD5', border: '#FDBA74', text: '#9A3412', icon: <AlertCircle {...({color: "#EA580C"} as any)} size={32} /> };
      return { bg: '#FEE2E2', border: '#FCA5A5', text: '#991B1B', icon: <AlertCircle {...({color: "#DC2626"} as any)} size={32} /> };
  };

  const styleConfig = feedback ? getFeedbackStyles(feedback.score) : null;

  return (
    <View style={styles.container}>
       <View style={styles.header}>
        <TouchableOpacity onPress={onExit} style={styles.backButton}>
          <ChevronLeft {...({color: "#4B5563"} as any)} size={24} />
          <Text style={styles.backText}>Back</Text>
        </TouchableOpacity>
        <View style={styles.titleContainer}>
            <Text style={styles.difficultyLabel}>{difficulty} Difficulty</Text>
            <Text style={styles.levelLabel}>Level {currentIndex + 1}/40</Text>
        </View>
        <View style={{width: 60}}></View>
      </View>

      <View style={styles.card}>
        
        <TouchableOpacity 
          onPress={prevWord} 
          disabled={currentIndex === 0}
          style={[styles.navButton, styles.navLeft, currentIndex === 0 && {opacity: 0.3}]}
        >
          <ChevronLeft size={24} {...({color: "#000"} as any)} />
        </TouchableOpacity>
        
        <TouchableOpacity 
          onPress={nextWord} 
          disabled={currentIndex === items.length - 1 || currentIndex >= unlockedLevel}
          style={[
            styles.navButton, 
            styles.navRight, 
            (currentIndex === items.length - 1 || currentIndex >= unlockedLevel) && {opacity: 0.3, backgroundColor: '#F3F4F6'}
          ]}
        >
          <ChevronRight size={24} {...({color: currentIndex >= unlockedLevel ? "#9CA3AF" : "#000"} as any)} />
        </TouchableOpacity>

        <ScrollView 
          contentContainerStyle={styles.scrollContent}
          showsVerticalScrollIndicator={false}
          style={{ width: '100%' }}
        >
            <View style={styles.wordContainer}>
              <Text style={styles.wordText}>{activeWord}</Text>
              <View style={styles.sentenceBox}>
                 <Text style={styles.sentenceText}>{activeSentence}</Text>
              </View>
            </View>

            <View style={styles.audioButtonsRow}>
               <TouchableOpacity 
                onPress={playWord}
                style={[styles.audioButton, {backgroundColor: '#EFF6FF'}]}
              >
                <View style={[styles.iconCircle, {backgroundColor: '#3B82F6'}]}>
                  <Volume2 size={24} {...({color: "#FFF"} as any)} />
                </View>
                <Text style={[styles.audioButtonText, {color: '#1D4ED8'}]}>Hear Word</Text>
              </TouchableOpacity>

              <TouchableOpacity 
                onPress={playSentence}
                style={[styles.audioButton, {backgroundColor: '#FAF5FF'}]}
              >
                <View style={[styles.iconCircle, {backgroundColor: '#A855F7'}]}>
                  <Volume2 size={24} {...({color: "#FFF"} as any)} />
                </View>
                <Text style={[styles.audioButtonText, {color: '#7E22CE'}]}>Hear Sentence</Text>
              </TouchableOpacity>
            </View>

            <View style={styles.actionContainer}>
              {!recording ? (
                <TouchableOpacity 
                  onPress={startRecording}
                  disabled={isProcessing}
                  style={[styles.recordButton, {backgroundColor: '#66BB6A'}]}
                >
                  <Mic size={28} {...({color: "#FFF"} as any)} />
                  <Text style={styles.recordButtonText}>{isProcessing ? 'Checking...' : 'Tap to Read'}</Text>
                </TouchableOpacity>
              ) : (
                <TouchableOpacity 
                  onPress={stopRecording}
                  style={[styles.recordButton, {backgroundColor: '#EF4444'}]}
                  disabled={isProcessing}
                >
                  {isProcessing ? <ActivityIndicator color="#FFF" /> : <Square size={28} {...({color: "#FFF"} as any)} {...({fill: "#FFF"} as any)} />}
                  <Text style={styles.recordButtonText}>{isProcessing ? 'Processing...' : 'Stop Recording'}</Text>
                </TouchableOpacity>
              )}
            </View>

            {isProcessing && (
               <View style={styles.loadingRow}>
                 <ActivityIndicator size="small" color="#4A90E2" />
                 <Text style={styles.loadingText}>Analyzing audio (this may take a moment)...</Text>
               </View>
            )}

            {feedback && styleConfig && (
              <View style={[styles.feedbackBox, {backgroundColor: styleConfig.bg, borderColor: styleConfig.border}]}>
                <View style={styles.feedbackHeader}>
                  <View style={styles.feedbackRow}>
                    {styleConfig.icon}
                    <Text style={[styles.feedbackTitle, {color: styleConfig.text}]}>{feedback.text}</Text>
                  </View>
                  <View style={styles.scoreBadge}>
                      <Text style={[styles.scoreText, {color: styleConfig.text}]}>{feedback.score}%</Text>
                  </View>
                </View>
                
                {feedback.transcript && (
                   <View style={styles.transcriptBox}>
                      <View style={styles.transcriptLabelRow}>
                         <MessageSquare size={12} {...({color: "#6B7280"} as any)} />
                         <Text style={styles.transcriptLabel}>You said:</Text>
                      </View>
                      <Text style={styles.transcriptText}>"{feedback.transcript}"</Text>
                   </View>
                )}

                {feedback.breakdown && (
                  <View style={styles.phoneticsBox}>
                      <Text style={styles.phoneticsText}>Phonetics: {feedback.breakdown}</Text>
                  </View>
                )}
                 {currentIndex < items.length - 1 && feedback.score > 60 && (
                     <TouchableOpacity onPress={nextWord} style={styles.nextLink}>
                        <Text style={styles.nextLinkText}>Next Word</Text>
                        <ArrowRight size={20} {...({color: "#4A90E2"} as any)} />
                     </TouchableOpacity>
                 )}
              </View>
            )}
        </ScrollView>
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
  },
  backText: {
    color: '#4B5563',
    fontWeight: 'bold',
    marginLeft: 4,
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
    alignItems: 'center',
    flex: 1,
    position: 'relative',
    borderWidth: 4,
    borderColor: '#E8F5E9',
    overflow: 'hidden', 
  },
  scrollContent: {
    padding: 24, 
    alignItems: 'center',
    flexGrow: 1,
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
  wordContainer: {
    alignItems: 'center',
    marginTop: 32,
    marginBottom: 48,
    width: '100%',
    paddingHorizontal: 32,
  },
  wordText: {
    fontSize: 48,
    fontWeight: 'bold',
    color: '#4A90E2',
    marginBottom: 32,
    textAlign: 'center',
  },
  sentenceBox: {
    backgroundColor: '#EFF6FF',
    paddingHorizontal: 24,
    paddingVertical: 16,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#DBEAFE',
    width: '100%',
  },
  sentenceText: {
    fontSize: 20,
    color: '#000000',
    fontWeight: 'bold',
    textAlign: 'center',
  },
  audioButtonsRow: {
    flexDirection: 'row',
    gap: 16,
    marginBottom: 48,
    justifyContent: 'center',
  },
  audioButton: {
    alignItems: 'center',
    justifyContent: 'center',
    gap: 8,
    padding: 12,
    borderRadius: 16,
    flex: 1,
    maxWidth: 140,
  },
  iconCircle: {
    padding: 12,
    borderRadius: 999,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.1,
    shadowRadius: 4,
    elevation: 3,
  },
  audioButtonText: {
    fontWeight: '600',
    fontSize: 14,
  },
  actionContainer: {
    marginBottom: 32,
    width: '100%',
    maxWidth: 320,
    alignItems: 'center',
  },
  recordButton: {
    width: '100%',
    paddingVertical: 20,
    borderRadius: 999,
    flexDirection: 'row',
    alignItems: 'center',
    justifyContent: 'center',
    gap: 12,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
  },
  recordButtonText: {
    color: '#FFFFFF',
    fontSize: 20,
    fontWeight: 'bold',
  },
  loadingRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginTop: 16,
  },
  loadingText: {
    color: '#6B7280',
    fontWeight: '500',
  },
  feedbackBox: {
    width: '100%',
    padding: 24,
    borderRadius: 16,
    borderWidth: 2,
    marginTop: 'auto',
    marginBottom: 16,
  },
  feedbackHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start', // Changed from center to allow multiline text alignment
    marginBottom: 16,
  },
  feedbackRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    flex: 1, // Allow row to take available width
    paddingRight: 12, // Add buffer between text and score badge
  },
  feedbackTitle: {
    fontSize: 22,
    fontWeight: 'bold',
    flexWrap: 'wrap', // Ensure text wraps
    flexShrink: 1, // Ensure text shrinks if needed
  },
  scoreBadge: {
    backgroundColor: 'rgba(255,255,255,0.5)',
    paddingHorizontal: 12,
    paddingVertical: 4,
    borderRadius: 8,
  },
  scoreText: {
    fontSize: 20,
    fontWeight: 'bold',
  },
  transcriptBox: {
    marginBottom: 12,
    backgroundColor: 'rgba(255,255,255,0.7)',
    padding: 12,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: 'rgba(0,0,0,0.05)',
    width: '100%',
  },
  transcriptLabelRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
    marginBottom: 4,
  },
  transcriptLabel: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#6B7280',
    textTransform: 'uppercase',
  },
  transcriptText: {
    color: '#1F2937',
    fontStyle: 'italic',
    fontWeight: '500',
    flexWrap: 'wrap',
    flexShrink: 1, // Prevent overflow
  },
  phoneticsBox: {
    backgroundColor: 'rgba(255,255,255,0.5)',
    padding: 8,
    borderRadius: 8,
    width: '100%',
  },
  phoneticsText: {
    fontSize: 14,
    color: '#6B7280',
    fontFamily: Platform.OS === 'ios' ? 'Courier' : 'monospace',
    flexWrap: 'wrap',
    flexShrink: 1, // Prevent overflow
  },
  nextLink: {
    marginTop: 16,
    alignSelf: 'flex-end',
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  nextLinkText: {
    color: '#4A90E2',
    fontWeight: 'bold',
  },
});