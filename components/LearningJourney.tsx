import React from 'react';
import { View, Text, TouchableOpacity, ScrollView, StyleSheet } from 'react-native';
import { UserProfile, Difficulty } from '../types';
import { Trophy, Star, Lock, CheckCircle, BookOpen, PenTool, Type, X, Zap } from 'lucide-react-native';

interface JourneyProps {
  user: UserProfile;
  onExit: () => void;
}

export const LearningJourney: React.FC<JourneyProps> = ({ user, onExit }) => {
  const levels = [Difficulty.PROFOUND, Difficulty.SEVERE, Difficulty.MODERATE, Difficulty.MILD];
  const currentLevelIndex = levels.indexOf(user.assignedDifficulty);

  // Helper to calculate score purely based on Activity History (Practice)
  const calculateSkillMastery = (skillType: 'Reading' | 'Writing' | 'Spelling' | 'Memory') => {
      const history = user.progressHistory || [];
      const relevant = history.filter(h => {
          if (skillType === 'Reading') return h.activityType === 'Reading';
          if (skillType === 'Writing') return h.activityType === 'Tracing'; // Tracing builds writing
          if (skillType === 'Spelling') return h.activityType === 'Spelling';
          if (skillType === 'Memory') return h.activityType === 'Memory';
          return false;
      });

      // If no practice activities completed, mastery is 0% (Start of journey)
      if (relevant.length === 0) return 0;

      const practiceAvg = relevant.reduce((acc, curr) => acc + curr.score, 0) / relevant.length;
      return Math.round(practiceAvg);
  };

  const readingScore = calculateSkillMastery('Reading');
  const writingScore = calculateSkillMastery('Writing');
  const spellingScore = calculateSkillMastery('Spelling');
  const memoryScore = calculateSkillMastery('Memory');

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <View style={styles.header}>
        <Text style={styles.title}>Learning Journey</Text>
        <TouchableOpacity onPress={onExit} style={styles.closeButton}>
            <X size={24} {...({color: "#6B7280"} as any)} />
        </TouchableOpacity>
      </View>

      {/* Progress Bars */}
      <View style={styles.progressCard}>
        <Text style={styles.cardTitle}>Activity Mastery</Text>
        <Text style={styles.cardSubtitle}>Average scores from daily practice activities</Text>
        <View style={styles.skillRowContainer}>
             {/* Reading */}
             <View style={styles.skillRow}>
                 <View style={styles.labelGroup}>
                    <BookOpen size={18} {...({color: "#3B82F6"} as any)} />
                    <Text style={styles.skillLabel}>Reading</Text>
                 </View>
                 <View style={styles.progressBarBg}>
                     <View style={[styles.progressBarFill, {backgroundColor: '#3B82F6', width: `${readingScore}%`}]}></View>
                 </View>
                 <Text style={styles.scoreText}>{readingScore}%</Text>
             </View>

             {/* Writing */}
             <View style={styles.skillRow}>
                 <View style={styles.labelGroup}>
                    <PenTool size={18} {...({color: "#22C55E"} as any)} />
                    <Text style={styles.skillLabel}>Writing</Text>
                 </View>
                 <View style={styles.progressBarBg}>
                     <View style={[styles.progressBarFill, {backgroundColor: '#22C55E', width: `${writingScore}%`}]}></View>
                 </View>
                 <Text style={styles.scoreText}>{writingScore}%</Text>
             </View>

             {/* Spelling */}
             <View style={styles.skillRow}>
                 <View style={styles.labelGroup}>
                    <Type size={18} {...({color: "#A855F7"} as any)} />
                    <Text style={styles.skillLabel}>Spelling</Text>
                 </View>
                 <View style={styles.progressBarBg}>
                     <View style={[styles.progressBarFill, {backgroundColor: '#A855F7', width: `${spellingScore}%`}]}></View>
                 </View>
                 <Text style={styles.scoreText}>{spellingScore}%</Text>
             </View>

             {/* Memory */}
             <View style={styles.skillRow}>
                 <View style={styles.labelGroup}>
                    <Zap size={18} {...({color: "#F59E0B"} as any)} />
                    <Text style={styles.skillLabel}>Memory</Text>
                 </View>
                 <View style={styles.progressBarBg}>
                     <View style={[styles.progressBarFill, {backgroundColor: '#F59E0B', width: `${memoryScore}%`}]}></View>
                 </View>
                 <Text style={styles.scoreText}>{memoryScore}%</Text>
             </View>
        </View>
      </View>

      {/* Path */}
      <View style={styles.pathContainer}>
         {/* Vertical Dashed Line */}
         <View style={styles.verticalLine} />

        {levels.map((level, idx) => {
            const isCompleted = idx < currentLevelIndex;
            const isCurrent = idx === currentLevelIndex;
            const isLocked = idx > currentLevelIndex;

            return (
                <View key={level} style={[
                    styles.levelCard, 
                    isCurrent && styles.levelCardCurrent,
                    isLocked && styles.levelCardLocked
                ]}>
                    {/* Node on line */}
                    <View style={[
                        styles.nodeCircle, 
                        isCurrent ? {borderColor: '#4A90E2'} : isCompleted ? {borderColor: '#10B981'} : {borderColor: '#D1D5DB'}
                    ]}>
                        {isCompleted ? <CheckCircle size={16} {...({color: "#10B981"} as any)} /> : isCurrent ? <Star size={16} {...({color: "#4A90E2"} as any)} {...({fill: "#4A90E2"} as any)} /> : <Lock size={14} {...({color: "#D1D5DB"} as any)} />}
                    </View>

                    <View style={styles.levelContent}>
                        <View style={{flex: 1, marginRight: 16}}>
                            <Text style={styles.levelTitle}>{level} Support Level</Text>
                            <Text style={styles.levelDesc}>
                                {level === Difficulty.PROFOUND && "Foundational motor skills and basic recognition."}
                                {level === Difficulty.SEVERE && "Guided practice with high tolerance."}
                                {level === Difficulty.MODERATE && "Building independence and accuracy."}
                                {level === Difficulty.MILD && "Refining skills for excellent performance."}
                            </Text>
                        </View>
                        {isCurrent && (
                            <View style={styles.currentBadge}>
                                <Text style={styles.currentBadgeText}>Current</Text>
                            </View>
                        )}
                    </View>
                </View>
            )
        })}
      </View>
    </ScrollView>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#FDFBF7',
  },
  scrollContent: {
    padding: 24,
    paddingBottom: 40,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 32,
  },
  title: {
    fontSize: 30,
    fontWeight: 'bold',
    color: '#2D2D2D',
  },
  closeButton: {
    padding: 8,
  },
  progressCard: {
    backgroundColor: '#FFFFFF',
    padding: 24,
    borderRadius: 24,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    marginBottom: 40,
    borderWidth: 1,
    borderColor: '#F3F4F6',
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 4,
    color: '#374151',
  },
  cardSubtitle: {
    fontSize: 12,
    color: '#9CA3AF',
    marginBottom: 24,
  },
  skillRowContainer: {
    gap: 24,
  },
  skillRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 16,
  },
  labelGroup: {
    width: 96,
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  skillLabel: {
    fontWeight: 'bold',
    color: '#4B5563',
  },
  progressBarBg: {
    flex: 1,
    height: 16,
    backgroundColor: '#F3F4F6',
    borderRadius: 999,
    overflow: 'hidden',
  },
  progressBarFill: {
    height: '100%',
  },
  scoreText: {
    fontSize: 14,
    fontWeight: 'bold',
    width: 40,
    textAlign: 'right',
    color: '#1F2937',
  },
  pathContainer: {
    paddingLeft: 24,
    position: 'relative',
  },
  verticalLine: {
    position: 'absolute',
    left: 26,
    top: 16,
    bottom: 16,
    width: 2,
    backgroundColor: '#D1D5DB',
  },
  levelCard: {
    position: 'relative',
    padding: 24,
    borderRadius: 16,
    borderWidth: 2,
    marginBottom: 32,
    backgroundColor: '#F9FAFB',
    borderColor: '#E5E7EB',
  },
  levelCardCurrent: {
    backgroundColor: '#FFFFFF',
    borderColor: '#4A90E2',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.1,
    shadowRadius: 20,
    elevation: 5,
  },
  levelCardLocked: {
    opacity: 0.6,
  },
  nodeCircle: {
    position: 'absolute',
    left: -40,
    top: '50%',
    marginTop: -16,
    width: 32,
    height: 32,
    borderRadius: 999,
    borderWidth: 4,
    alignItems: 'center',
    justifyContent: 'center',
    backgroundColor: '#FFFFFF',
    zIndex: 10,
  },
  levelContent: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  levelTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  levelDesc: {
    fontSize: 14,
    color: '#6B7280',
    marginTop: 4,
    lineHeight: 20,  },
  currentBadge: {
    paddingHorizontal: 12,
    paddingVertical: 4,
    backgroundColor: '#DBEAFE',
    borderRadius: 999,
  },
  currentBadgeText: {
    color: '#1D4ED8',
    fontSize: 12,
    fontWeight: 'bold',
    textTransform: 'uppercase',
  },
});