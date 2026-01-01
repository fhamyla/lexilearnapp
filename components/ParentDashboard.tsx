
import React, { useState } from 'react';
import { View, Text, ScrollView, TouchableOpacity, TextInput, Dimensions, StyleSheet } from 'react-native';
import { Svg, Path, Circle, Line, G, Text as SvgText } from 'react-native-svg';
import { ProgressRecord, Difficulty, AssessmentScores } from '../types';
import { Edit2, Save, Settings, Shield, BookOpen, PenTool, Type, LogOut, Zap, PlusCircle } from 'lucide-react-native';

interface DashboardProps {
  progressData: ProgressRecord[];
  childName: string;
  currentDifficulty: Difficulty;
  assessmentScores?: AssessmentScores;
  onUpdateChildName: (name: string) => void;
  onExit: () => void;
}

export const ParentDashboard: React.FC<DashboardProps> = ({ progressData, childName, currentDifficulty, assessmentScores, onUpdateChildName, onExit }) => {
  const [isEditingName, setIsEditingName] = useState(false);
  const [tempName, setTempName] = useState(childName);

  const handleSaveName = () => {
    onUpdateChildName(tempName);
    setIsEditingName(false);
  };

  const data = progressData || [];
  const hasData = data.length > 0;

  const avgScore = hasData 
    ? Math.round(data.reduce((acc: number, curr: ProgressRecord) => acc + curr.score, 0) / data.length) 
    : 0;
    
  const screenWidth = Dimensions.get('window').width;

  // Simple Line Chart Logic
  const chartHeight = 220;
  const chartWidth = screenWidth - 80; // Padding
  const maxScore = 100;
  const xGap = data.length > 1 ? chartWidth / (data.length - 1) : chartWidth;

  const points = data.map((d, i) => {
      const x = i * xGap;
      const y = chartHeight - (d.score / maxScore) * chartHeight;
      return { x, y, val: d.score, label: d.date };
  });

  const pathData = points.length > 1 
    ? points.map((p, i) => (i === 0 ? `M ${p.x} ${p.y}` : `L ${p.x} ${p.y}`)).join(' ')
    : points.length === 1 ? `M 0 ${points[0].y} L ${chartWidth} ${points[0].y}` : '';

  // Focus Area Logic
  let readingAvg = 0, writingAvg = 0, spellingAvg = 0;
  let focusAreas = [];

  if (assessmentScores) {
    readingAvg = Math.round((assessmentScores.readingSpeed + assessmentScores.readingAccuracy + assessmentScores.readingComprehension) / 3);
    writingAvg = Math.round((assessmentScores.writingSpeed + assessmentScores.writingQuality + assessmentScores.grammar) / 3);
    spellingAvg = Math.round((assessmentScores.phoneticSpelling + assessmentScores.irregularSpelling + assessmentScores.spellingAccuracy) / 3);
    
    const areas = [
        { name: 'Reading', score: readingAvg },
        { name: 'Writing', score: writingAvg },
        { name: 'Spelling', score: spellingAvg }
    ];

    if (assessmentScores.focusAreas && assessmentScores.focusAreas.length > 0) {
        // Use the saved AI-predicted order
        focusAreas = assessmentScores.focusAreas.map(name => {
            const match = areas.find(a => a.name === name);
            return match || { name, score: 0 };
        });
    } else {
        // Fallback: Sort ascending (lowest score is primary focus)
        focusAreas = areas.sort((a, b) => a.score - b.score);
    }
  } else {
    focusAreas = [];
  }

  // Determine Memory Score from Activity History
  const memoryRecords = data.filter(p => p.activityType === 'Memory');
  const memoryAvg = memoryRecords.length > 0 
      ? Math.round(memoryRecords.reduce((acc, curr) => acc + curr.score, 0) / memoryRecords.length) 
      : 0;

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.scrollContent}>
      <View style={styles.header}>
        <View style={{flex: 1, marginRight: 16}}>
            {isEditingName ? (
                <View style={styles.editNameRow}>
                    <TextInput 
                      value={tempName} 
                      onChangeText={setTempName}
                      style={styles.nameInput}
                      autoFocus
                    />
                    <TouchableOpacity onPress={handleSaveName} style={styles.saveButton}>
                        <Save size={20} {...({color: "#15803D"} as any)} />
                    </TouchableOpacity>
                </View>
            ) : (
                <TouchableOpacity onPress={() => setIsEditingName(true)} style={styles.editNameRow}>
                    <Text style={styles.headerTitle}>{childName}'s Progress</Text>
                    <Edit2 size={18} {...({color: "#9CA3AF"} as any)} />
                </TouchableOpacity>
            )}
        </View>
        
        <TouchableOpacity onPress={onExit} style={styles.exitButton}>
          <LogOut size={20} {...({color: "#374151"} as any)} />
        </TouchableOpacity>
      </View>

      {/* Summary Cards */}
      <View style={styles.summaryContainer}>
        <View style={[styles.summaryCard, {borderLeftColor: '#3B82F6'}]}>
          <Text style={styles.summaryLabel}>Avg Activity Score</Text>
          <Text style={styles.summaryValue}>{avgScore}%</Text>
        </View>
        <View style={[styles.summaryCard, {borderLeftColor: '#10B981'}]}>
          <Text style={styles.summaryLabel}>Activities Done</Text>
          <Text style={styles.summaryValue}>{data.length}</Text>
        </View>
        <View style={[styles.summaryCard, {borderLeftColor: '#8B5CF6'}]}>
          <Text style={styles.summaryLabel}>Focus Area</Text>
          <Text style={[styles.summaryValue, {fontSize: 18}]} numberOfLines={1}>{focusAreas[0]?.name || "N/A"}</Text>
        </View>
        <View style={[styles.summaryCard, {borderLeftColor: '#F97316'}]}>
          <View style={styles.row}>
             <Settings size={12} {...({color: "#6B7280"} as any)} />
             <Text style={styles.summaryLabel}>Level</Text>
          </View>
          <Text style={[styles.summaryValue, {fontSize: 18}]}>{currentDifficulty}</Text>
        </View>
      </View>

      {/* Custom Chart */}
      <View style={styles.chartCard}>
          <Text style={styles.cardTitle}>Recent Activity Performance</Text>
          {hasData ? (
              <View style={{ height: 250, width: '100%' }}>
                <Svg height="100%" width="100%" viewBox={`-10 -10 ${chartWidth + 40} ${chartHeight + 50}`}>
                    {/* Grid Lines */}
                    {[0, 25, 50, 75, 100].map((val) => {
                        const y = chartHeight - (val / 100) * chartHeight;
                        return (
                            <G key={val}>
                                <Line x1="0" y1={y} x2={chartWidth} y2={y} stroke="#E5E7EB" strokeWidth="1" />
                                <SvgText x={chartWidth + 10} y={y + 4} fontSize="10" fill="#9CA3AF">{val}</SvgText>
                            </G>
                        );
                    })}
                    {pathData ? <Path d={pathData} fill="none" stroke="#4A90E2" strokeWidth="3" /> : null}
                    {points.map((p, i) => (
                        <G key={i}>
                            <Circle cx={p.x} cy={p.y} r="5" fill="#4A90E2" stroke="#FFF" strokeWidth="2" />
                            <SvgText x={p.x} y={chartHeight + 20} fontSize="10" fill="#6B7280" textAnchor="middle">{p.label}</SvgText>
                        </G>
                    ))}
                </Svg>
              </View>
          ) : (
             <View style={styles.emptyStateBox}>
                 <PlusCircle size={48} {...({color: "#D1D5DB"} as any)} />
                 <Text style={styles.emptyStateText}>No activity data recorded yet.</Text>
                 <Text style={styles.emptyStateSub}>Complete activities in the Student Area to see progress charts here.</Text>
             </View>
          )}
      </View>

      {/* Priorities */}
      {focusAreas.length > 0 && (
        <View style={styles.prioritiesCard}>
           <Text style={styles.cardTitle}>Focus Areas (From Assessment)</Text>
           <View style={styles.priorityRow}>
               <View style={styles.priorityItem}>
                   <Text style={styles.priorityLabel}>Primary Focus</Text>
                   <Text style={styles.priorityValue}>{focusAreas[0]?.name}</Text>
                   <Text style={styles.priorityScore}>{focusAreas[0]?.score}%</Text>
               </View>
               <View style={[styles.priorityItem, {borderLeftWidth: 1, borderRightWidth: 1, borderColor: '#F3F4F6'}]}>
                   <Text style={styles.priorityLabel}>Secondary Focus</Text>
                   <Text style={styles.priorityValue}>{focusAreas[1]?.name}</Text>
                   <Text style={styles.priorityScore}>{focusAreas[1]?.score}%</Text>
               </View>
               <View style={styles.priorityItem}>
                   <Text style={styles.priorityLabel}>Last Focus</Text>
                   <Text style={styles.priorityValue}>{focusAreas[2]?.name}</Text>
                   <Text style={styles.priorityScore}>{focusAreas[2]?.score}%</Text>
               </View>
           </View>
        </View>
      )}

      {/* Breakdown */}
      {assessmentScores && (
          <View style={styles.breakdownCard}>
             <Text style={styles.cardTitle}>Initial Assessment Results</Text>
             <View style={{gap: 24}}>
                <View>
                   <View style={styles.breakdownHeader}>
                       <View style={styles.row}>
                           <BookOpen size={18} {...({color: "#4B5563"} as any)} />
                           <Text style={styles.breakdownLabel}>Reading</Text>
                       </View>
                       <Text style={[styles.breakdownScore, {color: '#4A90E2'}]}>{readingAvg}%</Text>
                   </View>
                   <View style={styles.progressBarBg}>
                       <View style={[styles.progressBarFill, {backgroundColor: '#4A90E2', width: `${readingAvg}%`}]} />
                   </View>
                </View>

                <View>
                   <View style={styles.breakdownHeader}>
                       <View style={styles.row}>
                           <PenTool size={18} {...({color: "#4B5563"} as any)} />
                           <Text style={styles.breakdownLabel}>Writing</Text>
                       </View>
                       <Text style={[styles.breakdownScore, {color: '#10B981'}]}>{writingAvg}%</Text>
                   </View>
                   <View style={styles.progressBarBg}>
                       <View style={[styles.progressBarFill, {backgroundColor: '#10B981', width: `${writingAvg}%`}]} />
                   </View>
                </View>

                <View>
                   <View style={styles.breakdownHeader}>
                       <View style={styles.row}>
                           <Type size={18} {...({color: "#4B5563"} as any)} />
                           <Text style={styles.breakdownLabel}>Spelling</Text>
                       </View>
                       <Text style={[styles.breakdownScore, {color: '#8B5CF6'}]}>{spellingAvg}%</Text>
                   </View>
                   <View style={styles.progressBarBg}>
                       <View style={[styles.progressBarFill, {backgroundColor: '#8B5CF6', width: `${spellingAvg}%`}]} />
                   </View>
                </View>
             </View>
          </View>
      )}
      
      <View style={styles.insightsCard}>
        <Shield size={24} {...({color: "#2563EB"} as any)} />
        <View style={{flex: 1}}>
            <Text style={styles.insightsTitle}>AI Insights for {childName}</Text>
            {focusAreas.length > 0 ? (
                <Text style={styles.insightsText}>
                Based on the assessment, {childName} shows strength in {focusAreas[2]?.name} but needs more practice in {focusAreas[0]?.name}. 
                Difficulty is set to {currentDifficulty}.
                </Text>
            ) : (
                <Text style={styles.insightsText}>
                    Please complete the initial assessment to view AI-driven insights and focus areas.
                </Text>
            )}
        </View>
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
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 32,
  },
  editNameRow: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 8,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1F2937',
  },
  nameInput: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#1F2937',
    borderBottomWidth: 2,
    borderBottomColor: '#4A90E2',
    flex: 1,
    padding: 0,
  },
  saveButton: {
    padding: 8,
    backgroundColor: '#DCFCE7',
    borderRadius: 999,
  },
  exitButton: {
    padding: 12,
    backgroundColor: '#E5E7EB',
    borderRadius: 12,
  },
  summaryContainer: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 16,
    marginBottom: 32,
  },
  summaryCard: {
    backgroundColor: '#FFFFFF',
    padding: 16,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.05,
    shadowRadius: 4,
    elevation: 2,
    borderLeftWidth: 4,
    width: '47%',
  },
  summaryLabel: {
    fontSize: 12,
    fontWeight: 'bold',
    color: '#6B7280',
    textTransform: 'uppercase',
  },
  summaryValue: {
    fontSize: 30,
    fontWeight: 'bold',
    color: '#1F2937',
    marginTop: 4,
  },
  chartCard: {
    backgroundColor: '#FFFFFF',
    padding: 24,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    marginBottom: 32,
  },
  cardTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    color: '#374151',
    marginBottom: 24,
  },
  emptyStateBox: {
    alignItems: 'center',
    justifyContent: 'center',
    padding: 32,
    gap: 16,
  },
  emptyStateText: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#9CA3AF',
  },
  emptyStateSub: {
    textAlign: 'center',
    color: '#9CA3AF',
    lineHeight: 20,
  },
  row: {
    flexDirection: 'row',
    alignItems: 'center',
    gap: 4,
  },
  breakdownCard: {
    backgroundColor: '#FFFFFF',
    padding: 24,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    marginBottom: 32,
  },
  breakdownHeader: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    marginBottom: 8,
  },
  breakdownLabel: {
    fontWeight: 'bold',
    color: '#4B5563',
  },
  breakdownScore: {
    fontWeight: 'bold',
  },
  progressBarBg: {
    height: 8,
    backgroundColor: '#F3F4F6',
    borderRadius: 999,
    width: '100%',
  },
  progressBarFill: {
    height: '100%',
    borderRadius: 999,
  },
  insightsCard: {
    backgroundColor: '#EFF6FF',
    padding: 24,
    borderRadius: 16,
    borderWidth: 1,
    borderColor: '#DBEAFE',
    flexDirection: 'row',
    gap: 16,
    marginBottom: 40,
  },
  insightsTitle: {
    fontWeight: 'bold',
    color: '#1E40AF',
    marginBottom: 8,
  },
  insightsText: {
    color: '#1D4ED8',
    lineHeight: 20,
  },
  prioritiesCard: {
    backgroundColor: '#FFFFFF',
    padding: 24,
    borderRadius: 16,
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.1,
    shadowRadius: 8,
    elevation: 4,
    marginBottom: 32,
  },
  priorityRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  priorityItem: {
    flex: 1,
    alignItems: 'center',
  },
  priorityLabel: {
    fontSize: 12,
    color: '#6B7280',
    fontWeight: 'bold',
    marginBottom: 8,
  },
  priorityValue: {
    fontSize: 16,
    fontWeight: 'bold',
    color: '#1F2937',
    marginBottom: 4,
  },
  priorityScore: {
    fontSize: 14,
    color: '#4A90E2',
    fontWeight: 'bold',
  },
});
