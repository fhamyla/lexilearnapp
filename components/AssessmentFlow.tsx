
import React, { useState, useRef, useEffect } from 'react';
import { View, Text, TouchableOpacity, PanResponder, ScrollView, Alert, ActivityIndicator, StyleSheet } from 'react-native';
import { Brain, Volume2, Trophy, Star, Mic, Square, Eraser, CheckCircle, AlertCircle, TrendingUp, Sparkles, Activity } from 'lucide-react-native';
import { Difficulty, UserProfile, AssessmentScores } from '../types';
import { db, doc, setDoc, updateDoc } from '../firebaseConfig';
import { analyzeReadingAssessment } from '../services/modelRequest';
import { Audio } from 'expo-av';
import * as Speech from 'expo-speech';
import { Svg, Path, Circle as SvgCircle } from 'react-native-svg';

interface AssessmentFlowProps {
  user: UserProfile;
  onComplete: (updatedProfile: UserProfile) => void;
}

const TASKS = [
  'Reading Speed',
  'Reading Accuracy',
  'Reading Comprehension',
  'Writing Speed',
  'Writing Quality',
  'Grammar & Sentence',
  'Phonetic Spelling',
  'Irregular Spelling',
  'Spelling Accuracy'
];

export const AssessmentFlow: React.FC<AssessmentFlowProps> = ({ user, onComplete }) => {
  const [currentTaskIndex, setCurrentTaskIndex] = useState(0);
  const [scores, setScores] = useState<number[]>(new Array(9).fill(0));
  const [isProcessing, setIsProcessing] = useState(false);
  const [isCalculatingPath, setIsCalculatingPath] = useState(false);
  const [calculatedProfile, setCalculatedProfile] = useState<UserProfile | null>(null);
  
  const [subStep, setSubStep] = useState(0);
  const [gameState, setGameState] = useState<'INTRO' | 'ACTIVE' | 'FEEDBACK'>('INTRO');
  const [accumulatedData, setAccumulatedData] = useState<any[]>([]); 
  const startTimeRef = useRef<number>(0);

  const userAge = user.childAge ? parseFloat(user.childAge) : 8.0;
  const effectiveAge = isNaN(userAge) ? 8.0 : userAge;

  // Drawing
  const [currentPath, setCurrentPath] = useState<string>('');
  const [isDrawing, setIsDrawing] = useState(false);
  const [tracingPoints, setTracingPoints] = useState<{x:number, y:number}[]>([]);

  // Audio
  const [recording, setRecording] = useState<Audio.Recording | null>(null);
  const recordingRef = useRef<Audio.Recording | null>(null);

  // --- DATA ---
  const readingSpeedText = "The quick brown fox jumps over the lazy dog. The sun is shining bright today.";
  
  const readingAccData = [
    { target: 'Through', options: ['Thought', 'Though', 'Through', 'Tough'] },
    { target: 'Quiet', options: ['Quite', 'Quiet', 'Quit', 'Quick'] },
    { target: 'Angel', options: ['Angle', 'Angel', 'Angry', 'Ankle'] },
    { target: 'Desert', options: ['Dessert', 'Desert', 'Deserve', 'Depart'] },
    { target: 'Loose', options: ['Lose', 'Loose', 'Loss', 'Lost'] }
  ];
  
  const readingCompPassage = "Sam has a red bike. He rides it to the park every day after school. His dog, Spot, runs beside him.";
  const readingCompData = [
    { q: "What color is Sam's bike?", a: "Red", options: ["Blue", "Red", "Green"] },
    { q: "Where does Sam go?", a: "The Park", options: ["School", "The Park", "Home"] },
    { q: "Who runs with Sam?", a: "Spot", options: ["Sam's Dad", "Spot", "A Cat"] }
  ];
  
  const writingSpeedTarget = "apple";
  const tracePath = "M 200,100 L 200,200 M 200,150 A 50,50 0 1,0 100,150 A 50,50 0 1,0 200,150"; 
  const traceViewBox = "0 0 300 300";
  
  const grammarData = [
    { q: "She ___ to the store.", a: "goes", options: ["go", "goes", "going"] },
    { q: "They ___ playing soccer.", a: "are", options: ["is", "am", "are"] },
    { q: "I ___ a book yesterday.", a: "read", options: ["read", "reads", "reading"] },
    { q: "The dog is ___ than the cat.", a: "bigger", options: ["big", "bigger", "biggest"] },
    { q: "___ is your name?", a: "What", options: ["When", "What", "Why"] }
  ];

  const phoneticData = [
    { target: 'Phone', options: ['Fone', 'Phone', 'Foan'] },
    { target: 'Laugh', options: ['Laff', 'Laugh', 'Lagh'] },
    { target: 'City', options: ['Sity', 'City', 'Citty'] }
  ];
  
  const irregularData = [
    { target: 'Wednesday', options: ['Wensday', 'Wednesday', 'Wednesdey'] },
    { target: 'People', options: ['Peeple', 'People', 'Poeple'] },
    { target: 'Knight', options: ['Nite', 'Knight', 'Night'] }
  ];

  const spellingAccData = [
    { target: 'Necessary', options: ['Neccessary', 'Necessary', 'Necesary'] },
    { target: 'Separate', options: ['Seperate', 'Separate', 'Sepperate'] },
    { target: 'Accommodate', options: ['Acommodate', 'Accomodate', 'Accommodate'] }
  ];

  const safeCleanupRecording = async (rec: Audio.Recording | null) => {
      if (!rec) return;
      try { await rec.stopAndUnloadAsync(); } catch (err) {}
  };

  useEffect(() => {
    setCurrentPath('');
    setTracingPoints([]);
  }, [currentTaskIndex]);

  useEffect(() => {
    return () => {
        if (recordingRef.current) {
            safeCleanupRecording(recordingRef.current);
            recordingRef.current = null;
        }
    };
  }, []);

  const panResponder = useRef(
    PanResponder.create({
      onStartShouldSetPanResponder: () => true,
      onPanResponderGrant: (evt: any) => {
        setIsDrawing(true);
        const { locationX, locationY } = evt.nativeEvent;
        setCurrentPath(prev => {
            const newPoint = `M ${locationX} ${locationY}`;
            return prev ? `${prev} ${newPoint}` : newPoint;
        });
        setTracingPoints(prev => [...prev, {x: locationX, y: locationY}]);
      },
      onPanResponderMove: (evt: any) => {
        const { locationX, locationY } = evt.nativeEvent;
        setCurrentPath(prev => `${prev} L ${locationX} ${locationY}`);
        setTracingPoints(prev => [...prev, {x: locationX, y: locationY}]);
      },
      onPanResponderRelease: () => {
        setIsDrawing(false);
      }
    })
  ).current;

  const startTask = () => {
    setGameState('ACTIVE');
    startTimeRef.current = Date.now();
  };

  const nextSubStep = () => {
    setSubStep(prev => prev + 1);
    startTimeRef.current = Date.now();
  };

  const playTTS = (text: string) => {
    Speech.speak(text, { rate: 0.9 });
  };

  const handleScoreAndNext = async (taskIdx: number, rawScore: number) => {
    if (recordingRef.current) { 
        await safeCleanupRecording(recordingRef.current);
        recordingRef.current = null;
        setRecording(null); 
    }

    const newScores = [...scores];
    newScores[taskIdx] = Math.max(0, Math.min(100, Math.round(rawScore)));
    setScores(newScores);
    setAccumulatedData([]);
    setTracingPoints([]);
    setSubStep(0);
    setIsProcessing(false);

    if (taskIdx < TASKS.length - 1) {
      setCurrentTaskIndex(prev => prev + 1);
      setGameState('INTRO');
    } else {
      finishAssessment(newScores);
    }
  };

  const handleReadingSpeedStart = async () => {
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
      setGameState('ACTIVE');
      startTimeRef.current = Date.now(); 
    } catch (err) {
      Alert.alert("Error", "Could not start microphone.");
    }
  };

  const handleReadingSpeedStop = async () => {
    if (isProcessing) return; 
    if (!recordingRef.current) return;

    setIsProcessing(true);
    const durationSec = (Date.now() - startTimeRef.current) / 1000;
    let uri: string | null = null;
    
    try {
        const status = await recordingRef.current.getStatusAsync();
        if (status.durationMillis < 1000) {
            Alert.alert("Too Short", "Please read for at least a second.");
            setIsProcessing(false);
            return;
        }
        await recordingRef.current.stopAndUnloadAsync();
        uri = recordingRef.current.getURI();
    } catch (e: any) {
        setIsProcessing(false);
        recordingRef.current = null;
        setRecording(null);
        return;
    }
    
    recordingRef.current = null;
    setRecording(null); 

    if (uri) {
       const result = await analyzeReadingAssessment(uri, readingSpeedText, durationSec);
       
       if (!result.transcript) {
          Alert.alert("Notice", "You didn't say anything");
          handleScoreAndNext(0, 50); 
          return;
       }
       
       let expectedWPM = 90;
       if (effectiveAge <= 7) expectedWPM = 60;
       else if (effectiveAge <= 9) expectedWPM = 90;
       else if (effectiveAge <= 11) expectedWPM = 120;
       else expectedWPM = 150;

       const wpmScore = Math.min(100, (result.wpm / expectedWPM) * 100);
       const finalScore = (result.accuracy * 0.7) + (wpmScore * 0.3);
       
       handleScoreAndNext(0, finalScore);
    } else {
       Alert.alert("Error", "Recording not found.");
       setIsProcessing(false);
    }
  };

  const handleWritingSpeedSubmit = () => {
    if (tracingPoints.length < 20) {
        Alert.alert("Incomplete", "Please write the word.");
        return;
    }
    const durationSec = (Date.now() - startTimeRef.current) / 1000;
    const minutes = durationSec / 60;
    const wpm = 1 / minutes; // 1 word / minutes
    
    // Scale expectation based on age (8-25 WPM)
    let expectedWPM = 15;
    if (effectiveAge <= 7) expectedWPM = 8;
    else if (effectiveAge <= 9) expectedWPM = 12;
    else if (effectiveAge <= 11) expectedWPM = 18;
    else expectedWPM = 25;

    let score = (wpm / expectedWPM) * 100;
    score = Math.min(100, Math.max(0, Math.round(score)));
    handleScoreAndNext(3, score);
  };

  const handleTraceQualitySubmit = () => {
     const points = tracingPoints.length;
     // Rubric Scoring:
     // Letter Formation (30) + Legibility (30) + Spacing (20) + Organization (20) = 100
     // We map "More tracing points" -> Better Formation/Legibility
     
     let formation = 0; // Max 30
     let legibility = 0; // Max 30
     const spacing = 20; // Max 20 (Assumed perfect for single isolated letter)
     const organization = 20; // Max 20 (Assumed perfect for single isolated letter)

     if (points > 80) { 
         formation = 30; 
         legibility = 30; 
     } else if (points > 50) { 
         formation = 20; 
         legibility = 20; 
     } else if (points > 20) { 
         formation = 10; 
         legibility = 10; 
     } else { 
         formation = 5; 
         legibility = 5; 
     }

     const totalScore = formation + legibility + spacing + organization;
     handleScoreAndNext(4, totalScore); 
  };

  const handleSelection = (correct: boolean, totalItems: number, idx: number) => {
    const newData = [...accumulatedData, correct];
    setAccumulatedData(newData);
    if (subStep < totalItems - 1) {
       nextSubStep();
    } else {
       const correctCount = newData.filter(Boolean).length;
       const score = Math.round((correctCount / totalItems) * 100);
       handleScoreAndNext(idx, score);
    }
  };

  const finishAssessment = async (finalScores: number[]) => {
    setIsCalculatingPath(true);
    
    // Default Fallback Calculation in case API fails
    const cleanScores = finalScores.map(s => isNaN(s) ? 0 : s);
    const readingAvg = Math.round((cleanScores[0] + cleanScores[1] + cleanScores[2]) / 3);
    const writingAvg = Math.round((cleanScores[3] + cleanScores[4] + cleanScores[5]) / 3);
    const spellingAvg = Math.round((cleanScores[6] + cleanScores[7] + cleanScores[8]) / 3);
    const overallAverage = Math.round((readingAvg + writingAvg + spellingAvg) / 3);

    let assignedDiff = Difficulty.MODERATE;
    if (overallAverage >= 80) assignedDiff = Difficulty.MILD;
    else if (overallAverage >= 60) assignedDiff = Difficulty.MODERATE;
    else if (overallAverage >= 40) assignedDiff = Difficulty.SEVERE;
    else assignedDiff = Difficulty.PROFOUND;
    
    const areas = [
        { name: 'Reading', score: readingAvg },
        { name: 'Writing', score: writingAvg },
        { name: 'Spelling', score: spellingAvg }
    ];
    // Sort by weakness (lowest score first)
    areas.sort((a, b) => a.score - b.score);
    let finalFocusOrder: string[] = areas.map(a => a.name);

    try {
        const payload = {
            age: effectiveAge || 8.0, 
            reading_speed: cleanScores[0],
            reading_accuracy: cleanScores[1],
            reading_comprehension: cleanScores[2],
            writing_speed: cleanScores[3],
            writing_quality: cleanScores[4],
            grammar_sentence: cleanScores[5],
            phonetic_spelling: cleanScores[6],
            irregular_word_spelling: cleanScores[7],
            spelling_accuracy: cleanScores[8]
        };

        const response = await fetch('https://lexilearnapp.onrender.com/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
            body: JSON.stringify(payload)
        });

        if (response.ok) {
            const data = await response.json();
            console.log("Prediction API Response:", JSON.stringify(data, null, 2));

            if (data.predicted_difficulty) {
                const diffLower = data.predicted_difficulty.toLowerCase();
                if (diffLower.includes('mild')) assignedDiff = Difficulty.MILD;
                else if (diffLower.includes('moderate')) assignedDiff = Difficulty.MODERATE;
                else if (diffLower.includes('severe')) assignedDiff = Difficulty.SEVERE;
                else if (diffLower.includes('profound')) assignedDiff = Difficulty.PROFOUND;
            }
            if (data.focus_areas && Array.isArray(data.focus_areas)) {
                // API returns ordered list: [Weakest (Primary), Secondary, Strongest (Last)]
                // Normalize names to Title Case to match areasMap keys later
                finalFocusOrder = data.focus_areas.map((f: any) => {
                    const name = f.name || '';
                    return name.charAt(0).toUpperCase() + name.slice(1).toLowerCase();
                });
            }
        }
    } catch (error: any) {
        console.warn("Prediction API failed/offline. Using local calculations.");
    }

    const assessmentScores: AssessmentScores = {
        readingSpeed: cleanScores[0], readingAccuracy: cleanScores[1], readingComprehension: cleanScores[2],
        writingSpeed: cleanScores[3], writingQuality: cleanScores[4], grammar: cleanScores[5],
        phoneticSpelling: cleanScores[6], irregularSpelling: cleanScores[7], spellingAccuracy: cleanScores[8],
        overallAverage,
        focusAreas: finalFocusOrder
    };

    const updatedProfile: UserProfile = { ...user, assessmentComplete: true, assignedDifficulty: assignedDiff, assessmentScores };

    setCalculatedProfile(updatedProfile);
    setIsCalculatingPath(false);

    if (user.uid) {
        try {
            // NOTE: We do NOT set assessmentComplete: true here yet. 
            // We wait for the user to click "Continue" on the results screen.
            // This prevents App.tsx from redirecting before the user sees the results.
            await setDoc(doc(db, "users", user.uid), { 
                assignedDifficulty: assignedDiff, 
                assessmentScores 
            }, { merge: true });
        } catch (e) {
            console.error("Error saving assessment results:", e);
        }
    }
  };

  const handleContinue = async () => {
      if (!calculatedProfile) return;
      
      // Finalize: Set assessmentComplete to true in Firestore
      if (user.uid) {
        try {
            await updateDoc(doc(db, "users", user.uid), { 
                assessmentComplete: true
            });
        } catch (e) {
            console.error("Error finalizing assessment:", e);
        }
      }
      
      onComplete(calculatedProfile);
  };

  const renderTask = () => {
      switch(currentTaskIndex) {
          case 0: return (
             <View style={styles.centerCol}>
                <Text style={styles.instructionText}>Read Aloud</Text>
                <View style={styles.passageContainer}>
                    <TouchableOpacity onPress={() => playTTS(readingSpeedText)} style={styles.audioButton}>
                        <Volume2 size={24} {...({color: "#2563EB"} as any)} />
                    </TouchableOpacity>
                    <Text style={styles.passageText}>{readingSpeedText}</Text>
                </View>
                {recording ? (
                    <TouchableOpacity onPress={handleReadingSpeedStop} style={styles.stopButton} disabled={isProcessing}>
                        {isProcessing ? <ActivityIndicator color="#FFF" /> : <Square {...({color: "#FFF"} as any)} {...({fill: "#FFF"} as any)} size={20} />}
                        <Text style={styles.buttonText}>{isProcessing ? "Analyzing..." : "Stop & Submit"}</Text>
                    </TouchableOpacity>
                ) : isProcessing ? (
                    <ActivityIndicator size="large" color="#4A90E2" />
                ) : <Text style={styles.hintText}>Tap microphone to start reading...</Text>}
             </View>
          );
          case 3: return (
             <View style={styles.centerCol}>
                <Text style={styles.instructionText}>Write the word:</Text>
                <Text style={styles.targetWord}>{writingSpeedTarget}</Text>
                <View style={styles.drawArea} {...panResponder.panHandlers}>
                   <Svg height="100%" width="100%">
                      <Path d={currentPath} stroke="#4A90E2" strokeWidth="4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                   </Svg>
                   <TouchableOpacity onPress={() => {setCurrentPath(''); setTracingPoints([])}} style={styles.clearButton}>
                      <Eraser size={20} {...({color: "#000"} as any)} />
                   </TouchableOpacity>
                </View>
                <TouchableOpacity onPress={handleWritingSpeedSubmit} style={styles.primaryButton}>
                   <Text style={styles.buttonText}>Done Writing</Text>
                </TouchableOpacity>
             </View>
          );
          case 4: return (
            <View style={styles.centerCol}>
               <Text style={styles.instructionText}>Trace the letter 'a' for Quality Check</Text>
               <View style={styles.drawArea} {...panResponder.panHandlers}>
                  <Svg height="100%" width="100%" viewBox={traceViewBox}>
                     <Path d={tracePath} stroke="#b0c4de" strokeWidth="10" fill="none" strokeLinecap="round" strokeLinejoin="round" strokeDasharray="15,15" opacity={0.8} />
                     <SvgCircle cx="200" cy="100" r="8" fill="#4A90E2" />
                     <SvgCircle cx="200" cy="150" r="8" fill="#4A90E2" />
                     <Path d={currentPath} stroke="#4A90E2" strokeWidth="4" fill="none" strokeLinecap="round" strokeLinejoin="round" />
                  </Svg>
                  <TouchableOpacity onPress={() => {setCurrentPath(''); setTracingPoints([])}} style={styles.clearButton}>
                     <Eraser size={20} {...({color: "#000"} as any)} />
                  </TouchableOpacity>
               </View>
               <TouchableOpacity onPress={handleTraceQualitySubmit} style={styles.primaryButton}>
                  <Text style={styles.buttonText}>Done Tracing</Text>
               </TouchableOpacity>
            </View>
          );
          default: 
            let data: any[] = [];
            let question = "";
            let options: string[] = [];
            let showAudio = false;
            let audioTarget = "";
            
            if (currentTaskIndex === 1) {
                data = readingAccData;
                question = `Find: "${data[subStep].target}"`;
                options = data[subStep].options;
                showAudio = true;
                audioTarget = data[subStep].target;
            } else if (currentTaskIndex === 2) {
                data = readingCompData;
                question = data[subStep].q;
                options = data[subStep].options;
            } else if (currentTaskIndex === 5) {
                data = grammarData;
                question = data[subStep].q.replace('___', '_____');
                options = data[subStep].options;
            } else if (currentTaskIndex === 6) {
                data = phoneticData;
                question = "Which is correct?";
                options = data[subStep].options;
                showAudio = true;
                audioTarget = data[subStep].target;
            } else if (currentTaskIndex === 7) {
                data = irregularData;
                question = "Which is correct?";
                options = data[subStep].options;
                showAudio = true;
                audioTarget = data[subStep].target;
            } else if (currentTaskIndex === 8) {
                data = spellingAccData;
                question = "Which is correct?";
                options = data[subStep].options;
                showAudio = true;
                audioTarget = data[subStep].target;
            }

            if (data.length > 0) {
                return (
                    <View style={styles.centerCol}>
                        {currentTaskIndex === 2 && (
                            <View style={[styles.passageContainer, {backgroundColor: '#EFF6FF', borderColor: '#DBEAFE'}]}>
                                <TouchableOpacity onPress={() => playTTS(readingCompPassage)} style={styles.audioButtonSmall}><Volume2 size={20} {...({color: "#4A90E2"} as any)} /></TouchableOpacity>
                                <Text style={styles.passageTextSmall}>{readingCompPassage}</Text>
                            </View>
                        )}
                        <Text style={styles.questionText}>{question}</Text>
                        {showAudio && (
                            <TouchableOpacity onPress={() => playTTS(audioTarget)} style={[styles.audioButton, {position: 'relative', marginBottom: 24, top: 0, right: 0}]}>
                                <Volume2 size={24} {...({color: "#2563EB"} as any)} />
                                <Text style={{marginLeft: 8, color: '#2563EB', fontWeight: 'bold'}}>Hear Word</Text>
                            </TouchableOpacity>
                        )}
                        <View style={styles.optionsContainer}>
                            {options.map((opt, i) => (
                                <TouchableOpacity 
                                    key={i} 
                                    onPress={() => {
                                        let correct = false;
                                        if (currentTaskIndex === 1) correct = opt === readingAccData[subStep].target;
                                        if (currentTaskIndex === 2) correct = opt === readingCompData[subStep].a;
                                        if (currentTaskIndex === 5) correct = opt === grammarData[subStep].a;
                                        if (currentTaskIndex === 6) correct = opt === phoneticData[subStep].target;
                                        if (currentTaskIndex === 7) correct = opt === irregularData[subStep].target;
                                        if (currentTaskIndex === 8) correct = opt === spellingAccData[subStep].target;
                                        handleSelection(correct, data.length, currentTaskIndex);
                                    }} 
                                    style={styles.optionButton}
                                >
                                    <Text style={styles.optionText}>{opt}</Text>
                                </TouchableOpacity>
                            ))}
                        </View>
                    </View>
                );
            }
            return <View><Text>Task {currentTaskIndex}</Text><TouchableOpacity onPress={() => handleScoreAndNext(currentTaskIndex, 100)}><Text>Skip (Demo)</Text></TouchableOpacity></View>;
      }
  }
  
  if (isCalculatingPath) {
      return (
          <View style={styles.loadingContainer}>
              <View style={styles.loadingCard}>
                  <View style={styles.loadingIcon}>
                    <Sparkles size={64} {...({color: "#4A90E2"} as any)} />
                  </View>
                  <Text style={styles.loadingTitle}>Calculating Learning Path...</Text>
                  <Text style={styles.loadingDesc}>Model is analyzing your performance and predicting the optimal difficulty level.</Text>
                  <ActivityIndicator size="large" color="#4A90E2" style={{marginTop: 32}} />
              </View>
          </View>
      );
  }

  if (calculatedProfile && calculatedProfile.assessmentScores) {
     const scores = calculatedProfile.assessmentScores;
     
     // Recalculate averages for display
     const readingAvg = Math.round((scores.readingSpeed + scores.readingAccuracy + scores.readingComprehension) / 3);
     const writingAvg = Math.round((scores.writingSpeed + scores.writingQuality + scores.grammar) / 3);
     const spellingAvg = Math.round((scores.phoneticSpelling + scores.irregularSpelling + scores.spellingAccuracy) / 3);
     
     const areasMap: Record<string, number> = {
         'Reading': readingAvg,
         'Writing': writingAvg,
         'Spelling': spellingAvg
     };

     let focusAreas;
     if (scores.focusAreas && scores.focusAreas.length > 0) {
         // Use the order from the API (names normalized in finishAssessment)
         focusAreas = scores.focusAreas.map(name => ({
             name,
             score: areasMap[name] || 0
         }));
     } else {
         focusAreas = [
            { name: 'Reading', score: readingAvg },
            { name: 'Writing', score: writingAvg },
            { name: 'Spelling', score: spellingAvg }
         ];
         focusAreas.sort((a, b) => a.score - b.score);
     }

     return (
        <ScrollView contentContainerStyle={styles.resultScroll} style={styles.container}>
             <View style={styles.resultCard}>
                 <View style={styles.trophyIcon}>
                    <Trophy size={48} {...({color: "#CA8A04"} as any)} />
                 </View>
                 <Text style={styles.resultTitle}>Calculated Learning Path!</Text>
                 <Text style={styles.resultSubtitle}>Here is your personalized learning plan</Text>
                 
                 <View style={styles.resultLevelBox}>
                    <Text style={styles.levelLabel}>Recommended Difficulty Level</Text>
                    <View style={styles.levelRow}>
                        <Text style={styles.levelText}>{calculatedProfile.assignedDifficulty}</Text>
                        <Star {...({fill: "#4A90E2"} as any)} {...({color: "#4A90E2"} as any)} size={32} />
                    </View>
                 </View>

                 <View style={styles.focusContainer}>
                    <Text style={styles.focusHeader}>Predicted Focus Areas</Text>
                    
                    {/* PRIMARY FOCUS (Weakest) */}
                    <View style={[styles.focusItem, { backgroundColor: '#FEF2F2', borderColor: '#FECACA' }]}>
                        <View style={styles.focusRow}>
                            <AlertCircle {...({color: "#EF4444"} as any)} size={28} />
                            <View style={{flex: 1}}>
                                <Text style={[styles.focusLabel, {color: '#B91C1C'}]}>Primary Focus</Text>
                                <Text style={styles.focusValue}>{focusAreas[0]?.name || 'N/A'}</Text>
                            </View>
                            <View style={styles.scoreBadge}>
                                <Text style={[styles.focusScore, {color: '#B91C1C'}]}>{focusAreas[0]?.score || 0}%</Text>
                            </View>
                        </View>
                    </View>

                    {/* SECONDARY FOCUS */}
                    <View style={[styles.focusItem, { backgroundColor: '#FFFBEB', borderColor: '#FDE68A' }]}>
                        <View style={styles.focusRow}>
                            <Activity {...({color: "#D97706"} as any)} size={28} />
                            <View style={{flex: 1}}>
                                <Text style={[styles.focusLabel, {color: '#B45309'}]}>Secondary Focus</Text>
                                <Text style={styles.focusValue}>{focusAreas[1]?.name || 'N/A'}</Text>
                            </View>
                            <View style={styles.scoreBadge}>
                                <Text style={[styles.focusScore, {color: '#B45309'}]}>{focusAreas[1]?.score || 0}%</Text>
                            </View>
                        </View>
                    </View>

                    {/* LAST FOCUS (Strongest) */}
                    <View style={[styles.focusItem, { backgroundColor: '#F0FDF4', borderColor: '#86EFAC' }]}>
                        <View style={styles.focusRow}>
                            <CheckCircle {...({color: "#16A34A"} as any)} size={28} />
                            <View style={{flex: 1}}>
                                <Text style={[styles.focusLabel, {color: '#15803D'}]}>Last Focus</Text>
                                <Text style={styles.focusValue}>{focusAreas[2]?.name || 'N/A'}</Text>
                            </View>
                            <View style={styles.scoreBadge}>
                                <Text style={[styles.focusScore, {color: '#15803D'}]}>{focusAreas[2]?.score || 0}%</Text>
                            </View>
                        </View>
                    </View>
                 </View>

                 <TouchableOpacity 
                    onPress={handleContinue}
                    style={styles.primaryButton}
                 >
                    <Text style={styles.buttonText}>Continue to Student Dashboard</Text>
                 </TouchableOpacity>
             </View>
        </ScrollView>
     )
  }

  return (
    <ScrollView contentContainerStyle={styles.scrollContent} style={styles.container}>
      <View style={styles.progressRow}>
        <Text style={styles.progressLabel}>PROGRESS</Text>
        <Text style={styles.progressLabel}>{Math.round((currentTaskIndex / TASKS.length) * 100)}%</Text>
      </View>
      <View style={styles.progressBarBg}>
         <View style={[styles.progressBarFill, {width: `${(currentTaskIndex / TASKS.length) * 100}%`}]} />
      </View>

      <View style={styles.taskCard}>
         {gameState === 'INTRO' ? (
             <View style={styles.centerCol}>
                <View style={styles.iconCircle}>
                   <Brain size={48} {...({color: "#4A90E2"} as any)} />
                </View>
                <Text style={styles.taskTitle}>{TASKS[currentTaskIndex]}</Text>
                <Text style={styles.taskDesc}>
                    {currentTaskIndex === 0 ? "Read the text aloud." : "Complete the task."}
                </Text>
                
                {/* PREVIEW PASSAGE FOR READING TASK */}
                {currentTaskIndex === 0 && (
                    <View style={[styles.passageContainer, { marginBottom: 32 }]}>
                        <View style={{ flexDirection: 'row', justifyContent: 'center', alignItems: 'center', gap: 8, marginBottom: 12 }}>
                            <Text style={[styles.instructionText, { marginBottom: 0, textAlign: 'center' }]}>Challenge Text Preview</Text>
                            <TouchableOpacity onPress={() => playTTS(readingSpeedText)} style={{ backgroundColor: '#DBEAFE', padding: 6, borderRadius: 20 }}>
                                <Volume2 size={18} {...({color: "#2563EB"} as any)} />
                            </TouchableOpacity>
                        </View>
                        <Text style={[styles.passageText, { fontSize: 18, textAlign: 'center', marginTop: 0 }]}>{readingSpeedText}</Text>
                    </View>
                )}

                {currentTaskIndex === 0 ? (
                    <TouchableOpacity onPress={handleReadingSpeedStart} style={[styles.primaryButton, {flexDirection: 'row', gap: 12}]}>
                        <Mic {...({color: "#FFF"} as any)} />
                        <Text style={styles.buttonText}>Start Recording</Text>
                    </TouchableOpacity>
                ) : (
                    <TouchableOpacity onPress={startTask} style={styles.primaryButton}>
                        <Text style={styles.buttonText}>Start</Text>
                    </TouchableOpacity>
                )}
             </View>
         ) : renderTask()}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#FDFBF7' },
  scrollContent: { flexGrow: 1, padding: 16 },
  progressRow: { flexDirection: 'row', justifyContent: 'space-between', marginBottom: 8 },
  progressLabel: { fontSize: 14, fontWeight: 'bold', color: '#9CA3AF' },
  progressBarBg: { height: 12, backgroundColor: '#E5E7EB', borderRadius: 6, marginBottom: 32, overflow: 'hidden' },
  progressBarFill: { height: '100%', backgroundColor: '#4A90E2' },
  taskCard: { backgroundColor: '#FFFFFF', borderRadius: 24, padding: 24, borderWidth: 2, borderColor: '#EFF6FF', minHeight: 500, alignItems: 'center', justifyContent: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.1, shadowRadius: 8, elevation: 3 },
  centerCol: { alignItems: 'center', width: '100%' },
  iconCircle: { width: 96, height: 96, backgroundColor: '#DBEAFE', borderRadius: 48, alignItems: 'center', justifyContent: 'center', marginBottom: 24 },
  taskTitle: { fontSize: 24, fontWeight: 'bold', marginBottom: 16, textAlign: 'center', color: '#1F2937' },
  taskDesc: { color: '#6B7280', marginBottom: 32, textAlign: 'center', fontSize: 18 },
  primaryButton: { paddingHorizontal: 48, paddingVertical: 16, backgroundColor: '#4A90E2', borderRadius: 999, alignItems: 'center', justifyContent: 'center', width: '100%' },
  buttonText: { color: '#FFFFFF', fontWeight: 'bold', fontSize: 18 },
  instructionText: { color: '#6B7280', fontWeight: 'bold', marginBottom: 16, fontSize: 16 },
  passageContainer: { backgroundColor: '#FFFBEB', padding: 24, borderRadius: 12, borderWidth: 2, borderColor: '#FDE68A', marginBottom: 24, width: '100%', position: 'relative' },
  audioButton: { position: 'absolute', top: 8, right: 8, padding: 8, backgroundColor: '#DBEAFE', borderRadius: 999, flexDirection: 'row', alignItems: 'center' },
  passageText: { fontSize: 20, fontWeight: 'bold', color: '#000', marginTop: 16, lineHeight: 30 },
  stopButton: { width: '100%', paddingVertical: 16, backgroundColor: '#EF4444', borderRadius: 12, flexDirection: 'row', alignItems: 'center', justifyContent: 'center', gap: 8 },
  hintText: { color: '#6B7280', fontSize: 16 },
  targetWord: { fontSize: 48, fontWeight: 'bold', marginBottom: 24, color: '#4A90E2' },
  drawArea: { width: 300, height: 300, backgroundColor: '#FFFFFF', borderRadius: 12, borderWidth: 2, borderColor: '#D1D5DB', marginBottom: 24, overflow: 'hidden' },
  clearButton: { position: 'absolute', top: 8, right: 8, padding: 8, backgroundColor: '#F3F4F6', borderRadius: 999 },
  audioButtonSmall: { position: 'absolute', top: 8, right: 8, padding: 8 },
  passageTextSmall: { fontSize: 18, color: '#1F2937', paddingRight: 40 },
  questionText: { fontSize: 24, fontWeight: 'bold', marginBottom: 24, textAlign: 'center', color: '#1F2937' },
  optionsContainer: { width: '100%', gap: 12 },
  optionButton: { padding: 16, backgroundColor: '#FFFFFF', borderWidth: 2, borderColor: '#E5E7EB', borderRadius: 12 },
  optionText: { fontWeight: 'bold', fontSize: 18, color: '#374151', textAlign: 'center' },
  loadingContainer: { flex: 1, justifyContent: 'center', alignItems: 'center', backgroundColor: '#FDFBF7' },
  loadingCard: { backgroundColor: '#FFFFFF', padding: 32, borderRadius: 24, alignItems: 'center', shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.1, shadowRadius: 8, elevation: 4, width: '80%', maxWidth: 400 },
  loadingIcon: { marginBottom: 24 },
  loadingTitle: { fontSize: 20, fontWeight: 'bold', color: '#1F2937', textAlign: 'center', marginBottom: 8 },
  loadingDesc: { color: '#6B7280', textAlign: 'center', marginBottom: 24 },
  resultScroll: { flexGrow: 1, padding: 16, justifyContent: 'center' },
  resultCard: { backgroundColor: '#FFFFFF', padding: 24, borderRadius: 24, shadowColor: '#000', shadowOffset: { width: 0, height: 4 }, shadowOpacity: 0.1, shadowRadius: 8, elevation: 4, alignItems: 'center', width: '100%' },
  trophyIcon: { width: 80, height: 80, backgroundColor: '#FEF9C3', borderRadius: 999, alignItems: 'center', justifyContent: 'center', marginBottom: 24 },
  resultTitle: { fontSize: 28, fontWeight: 'bold', color: '#1F2937', textAlign: 'center', marginBottom: 8 },
  resultSubtitle: { fontSize: 16, color: '#6B7280', textAlign: 'center', marginBottom: 32 },
  resultLevelBox: { width: '100%', backgroundColor: '#EFF6FF', padding: 16, borderRadius: 16, borderWidth: 1, borderColor: '#DBEAFE', marginBottom: 32 },
  levelLabel: { fontSize: 12, fontWeight: 'bold', color: '#6B7280', textTransform: 'uppercase', marginBottom: 4 },
  levelRow: { flexDirection: 'row', alignItems: 'center', justifyContent: 'space-between' },
  levelText: { fontSize: 24, fontWeight: 'bold', color: '#1D4ED8' },
  focusContainer: { width: '100%', marginBottom: 32 },
  focusHeader: { fontSize: 18, fontWeight: 'bold', color: '#1F2937', marginBottom: 16 },
  focusItem: { borderWidth: 2, borderRadius: 16, padding: 16, marginBottom: 12 },
  focusRow: { flexDirection: 'row', alignItems: 'center', gap: 16 },
  focusLabel: { fontSize: 12, fontWeight: 'bold', textTransform: 'uppercase', marginBottom: 2 },
  focusValue: { fontSize: 18, fontWeight: 'bold', color: '#1F2937' },
  scoreBadge: { backgroundColor: '#FFFFFF', paddingHorizontal: 12, paddingVertical: 4, borderRadius: 8, shadowColor: '#000', shadowOffset: {width: 0, height: 1}, shadowOpacity: 0.05, shadowRadius: 2, elevation: 1 },
  focusScore: { fontSize: 16, fontWeight: 'bold' },
});
