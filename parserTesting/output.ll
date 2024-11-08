; ModuleID = 'tests.c'
source_filename = "tests.c"
target datalayout = "e-m:e-p270:32:32-p271:32:32-p272:64:64-i64:64-f80:128-n8:16:32:64-S128"
target triple = "x86_64-pc-linux-gnu"

@__const.simpleDeclarations.s = private unnamed_addr constant [10 x i8] c"Hello\00\00\00\00\00", align 1
@__const.declarationsWithAssignment.arr = private unnamed_addr constant [3 x i32] [i32 1, i32 2, i32 3], align 4
@__const.complexLoop.arr = private unnamed_addr constant [5 x i32] [i32 10, i32 20, i32 30, i32 40, i32 50], align 16
@.str = private unnamed_addr constant [21 x i8] c"Sum: %d\0AProduct: %d\0A\00", align 1
@__const.testPointerArithmetic.arr = private unnamed_addr constant [5 x i32] [i32 1, i32 2, i32 3, i32 4, i32 5], align 16
@__const.testPointerToArray.matrix = private unnamed_addr constant [3 x [3 x i32]] [[3 x i32] [i32 1, i32 2, i32 3], [3 x i32] [i32 4, i32 5, i32 6], [3 x i32] [i32 7, i32 8, i32 9]], align 16
@__const.testPointerDifference.numbers = private unnamed_addr constant [4 x i32] [i32 10, i32 20, i32 30, i32 40], align 16
@__const.testPointerToArrayElement.grid = private unnamed_addr constant [2 x [2 x i32]] [[2 x i32] [i32 1, i32 2], [2 x i32] [i32 3, i32 4]], align 16

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @simpleDeclarations() #0 {
  %1 = alloca i32, align 4
  %2 = alloca float, align 4
  %3 = alloca i8, align 1
  %4 = alloca [10 x i8], align 1
  %5 = alloca [5 x i32], align 16
  store i8 65, i8* %3, align 1
  %6 = bitcast [10 x i8]* %4 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 1 %6, i8* align 1 getelementptr inbounds ([10 x i8], [10 x i8]* @__const.simpleDeclarations.s, i32 0, i32 0), i64 10, i1 false)
  ret void
}

; Function Attrs: argmemonly nofree nounwind willreturn
declare void @llvm.memcpy.p0i8.p0i8.i64(i8* noalias nocapture writeonly, i8* noalias nocapture readonly, i64, i1 immarg) #1

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @declarationsWithAssignment() #0 {
  %1 = alloca i32, align 4
  %2 = alloca float, align 4
  %3 = alloca [3 x i32], align 4
  store i32 10, i32* %1, align 4
  store float 5.500000e+00, float* %2, align 4
  %4 = bitcast [3 x i32]* %3 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 4 %4, i8* align 4 bitcast ([3 x i32]* @__const.declarationsWithAssignment.arr to i8*), i64 12, i1 false)
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @basicForLoop() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  store i32 0, i32* %2, align 4
  br label %3

3:                                                ; preds = %10, %0
  %4 = load i32, i32* %2, align 4
  %5 = icmp slt i32 %4, 10
  br i1 %5, label %6, label %13

6:                                                ; preds = %3
  %7 = load i32, i32* %2, align 4
  %8 = load i32, i32* %1, align 4
  %9 = add nsw i32 %8, %7
  store i32 %9, i32* %1, align 4
  br label %10

10:                                               ; preds = %6
  %11 = load i32, i32* %2, align 4
  %12 = add nsw i32 %11, 1
  store i32 %12, i32* %2, align 4
  br label %3, !llvm.loop !6

13:                                               ; preds = %3
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @nestedForLoop() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  br label %4

4:                                                ; preds = %19, %0
  %5 = load i32, i32* %1, align 4
  %6 = icmp slt i32 %5, 3
  br i1 %6, label %7, label %22

7:                                                ; preds = %4
  store i32 0, i32* %2, align 4
  br label %8

8:                                                ; preds = %15, %7
  %9 = load i32, i32* %2, align 4
  %10 = icmp slt i32 %9, 2
  br i1 %10, label %11, label %18

11:                                               ; preds = %8
  %12 = load i32, i32* %1, align 4
  %13 = load i32, i32* %2, align 4
  %14 = mul nsw i32 %12, %13
  store i32 %14, i32* %3, align 4
  br label %15

15:                                               ; preds = %11
  %16 = load i32, i32* %2, align 4
  %17 = add nsw i32 %16, 1
  store i32 %17, i32* %2, align 4
  br label %8, !llvm.loop !8

18:                                               ; preds = %8
  br label %19

19:                                               ; preds = %18
  %20 = load i32, i32* %1, align 4
  %21 = add nsw i32 %20, 1
  store i32 %21, i32* %1, align 4
  br label %4, !llvm.loop !9

22:                                               ; preds = %4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @conditionalBlock(i32 noundef %0) #0 {
  %2 = alloca i32, align 4
  store i32 %0, i32* %2, align 4
  %3 = load i32, i32* %2, align 4
  %4 = add nsw i32 %3, 1
  %5 = icmp sgt i32 %4, 0
  br i1 %5, label %6, label %9

6:                                                ; preds = %1
  %7 = load i32, i32* %2, align 4
  %8 = sub nsw i32 %7, 1
  store i32 %8, i32* %2, align 4
  br label %10

9:                                                ; preds = %1
  store i32 0, i32* %2, align 4
  br label %10

10:                                               ; preds = %9, %6
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @binaryExpressions() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca i32, align 4
  store i32 10, i32* %3, align 4
  store i32 20, i32* %4, align 4
  %6 = load i32, i32* %3, align 4
  %7 = load i32, i32* %4, align 4
  %8 = mul nsw i32 %7, 2
  %9 = add nsw i32 %6, %8
  store i32 %9, i32* %5, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @complexExpressions(i32 noundef %0, i32 noundef %1, i32 noundef %2, i32 noundef %3) #0 {
  %5 = alloca i32, align 4
  %6 = alloca i32, align 4
  %7 = alloca i32, align 4
  %8 = alloca i32, align 4
  %9 = alloca i32, align 4
  store i32 %0, i32* %5, align 4
  store i32 %1, i32* %6, align 4
  store i32 %2, i32* %7, align 4
  store i32 %3, i32* %8, align 4
  %10 = load i32, i32* %5, align 4
  %11 = load i32, i32* %6, align 4
  %12 = add nsw i32 %10, %11
  %13 = load i32, i32* %7, align 4
  %14 = sub nsw i32 %13, 3
  %15 = mul nsw i32 %12, %14
  %16 = load i32, i32* %8, align 4
  %17 = add nsw i32 %16, 1
  %18 = sdiv i32 %15, %17
  store i32 %18, i32* %9, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @forLoopWithBreakContinue() #0 {
  %1 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  br label %2

2:                                                ; preds = %14, %0
  %3 = load i32, i32* %1, align 4
  %4 = icmp slt i32 %3, 10
  br i1 %4, label %5, label %17

5:                                                ; preds = %2
  %6 = load i32, i32* %1, align 4
  %7 = icmp eq i32 %6, 5
  br i1 %7, label %8, label %9

8:                                                ; preds = %5
  br label %14

9:                                                ; preds = %5
  %10 = load i32, i32* %1, align 4
  %11 = icmp eq i32 %10, 8
  br i1 %11, label %12, label %13

12:                                               ; preds = %9
  br label %17

13:                                               ; preds = %9
  br label %14

14:                                               ; preds = %13, %8
  %15 = load i32, i32* %1, align 4
  %16 = add nsw i32 %15, 1
  store i32 %16, i32* %1, align 4
  br label %2, !llvm.loop !10

17:                                               ; preds = %12, %2
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @whileLoops() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  store i32 0, i32* %1, align 4
  br label %3

3:                                                ; preds = %6, %0
  %4 = load i32, i32* %1, align 4
  %5 = icmp slt i32 %4, 5
  br i1 %5, label %6, label %9

6:                                                ; preds = %3
  %7 = load i32, i32* %1, align 4
  %8 = add nsw i32 %7, 1
  store i32 %8, i32* %1, align 4
  br label %3, !llvm.loop !11

9:                                                ; preds = %3
  store i32 5, i32* %2, align 4
  br label %10

10:                                               ; preds = %13, %9
  %11 = load i32, i32* %2, align 4
  %12 = add nsw i32 %11, -1
  store i32 %12, i32* %2, align 4
  br label %13

13:                                               ; preds = %10
  %14 = load i32, i32* %2, align 4
  %15 = icmp sgt i32 %14, 0
  br i1 %15, label %10, label %16, !llvm.loop !12

16:                                               ; preds = %13
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @complexLoop() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32, align 4
  %3 = alloca i32, align 4
  %4 = alloca i32, align 4
  %5 = alloca [5 x i32], align 16
  store i32 0, i32* %3, align 4
  store i32 1, i32* %4, align 4
  store i32 1, i32* %1, align 4
  br label %6

6:                                                ; preds = %34, %0
  %7 = load i32, i32* %1, align 4
  %8 = icmp sle i32 %7, 5
  br i1 %8, label %9, label %37

9:                                                ; preds = %6
  %10 = load i32, i32* %1, align 4
  %11 = load i32, i32* %3, align 4
  %12 = add nsw i32 %11, %10
  store i32 %12, i32* %3, align 4
  %13 = load i32, i32* %1, align 4
  %14 = load i32, i32* %4, align 4
  %15 = mul nsw i32 %14, %13
  store i32 %15, i32* %4, align 4
  %16 = load i32, i32* %1, align 4
  %17 = srem i32 %16, 2
  %18 = icmp eq i32 %17, 0
  br i1 %18, label %19, label %22

19:                                               ; preds = %9
  %20 = load i32, i32* %4, align 4
  %21 = mul nsw i32 %20, 2
  store i32 %21, i32* %4, align 4
  br label %25

22:                                               ; preds = %9
  %23 = load i32, i32* %4, align 4
  %24 = mul nsw i32 %23, 3
  store i32 %24, i32* %4, align 4
  br label %25

25:                                               ; preds = %22, %19
  %26 = bitcast [5 x i32]* %5 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %26, i8* align 16 bitcast ([5 x i32]* @__const.complexLoop.arr to i8*), i64 20, i1 false)
  %27 = load i32, i32* %1, align 4
  %28 = load i32, i32* %1, align 4
  %29 = sub nsw i32 %28, 1
  %30 = sext i32 %29 to i64
  %31 = getelementptr inbounds [5 x i32], [5 x i32]* %5, i64 0, i64 %30
  %32 = load i32, i32* %31, align 4
  %33 = add nsw i32 %32, %27
  store i32 %33, i32* %31, align 4
  br label %34

34:                                               ; preds = %25
  %35 = load i32, i32* %1, align 4
  %36 = add nsw i32 %35, 1
  store i32 %36, i32* %1, align 4
  br label %6, !llvm.loop !13

37:                                               ; preds = %6
  %38 = load i32, i32* %3, align 4
  %39 = load i32, i32* %4, align 4
  %40 = call i32 (i8*, ...) @printf(i8* noundef getelementptr inbounds ([21 x i8], [21 x i8]* @.str, i64 0, i64 0), i32 noundef %38, i32 noundef %39)
  ret void
}

declare i32 @printf(i8* noundef, ...) #2

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @test3DArray() #0 {
  %1 = alloca [2 x [3 x [4 x i32]]], align 16
  %2 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* %1, i64 0, i64 1
  %3 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %2, i64 0, i64 2
  %4 = getelementptr inbounds [4 x i32], [4 x i32]* %3, i64 0, i64 3
  store i32 42, i32* %4, align 4
  %5 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* %1, i64 0, i64 1
  %6 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %5, i64 0, i64 2
  %7 = getelementptr inbounds [4 x i32], [4 x i32]* %6, i64 0, i64 3
  %8 = load i32, i32* %7, align 4
  %9 = getelementptr inbounds [2 x [3 x [4 x i32]]], [2 x [3 x [4 x i32]]]* %1, i64 0, i64 0
  %10 = getelementptr inbounds [3 x [4 x i32]], [3 x [4 x i32]]* %9, i64 0, i64 0
  %11 = getelementptr inbounds [4 x i32], [4 x i32]* %10, i64 0, i64 0
  store i32 %8, i32* %11, align 16
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @testPointerDereference() #0 {
  %1 = alloca i32, align 4
  %2 = alloca i32*, align 8
  %3 = alloca i32**, align 8
  store i32 5, i32* %1, align 4
  store i32* %1, i32** %2, align 8
  store i32** %2, i32*** %3, align 8
  %4 = load i32*, i32** %2, align 8
  store i32 10, i32* %4, align 4
  %5 = load i32**, i32*** %3, align 8
  %6 = load i32*, i32** %5, align 8
  store i32 15, i32* %6, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @testPointerArithmetic() #0 {
  %1 = alloca [5 x i32], align 16
  %2 = alloca i32*, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32, align 4
  %5 = bitcast [5 x i32]* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %5, i8* align 16 bitcast ([5 x i32]* @__const.testPointerArithmetic.arr to i8*), i64 20, i1 false)
  %6 = getelementptr inbounds [5 x i32], [5 x i32]* %1, i64 0, i64 0
  store i32* %6, i32** %2, align 8
  %7 = load i32*, i32** %2, align 8
  %8 = getelementptr inbounds i32, i32* %7, i64 2
  store i32* %8, i32** %3, align 8
  %9 = load i32*, i32** %2, align 8
  %10 = getelementptr inbounds i32, i32* %9, i64 3
  %11 = load i32, i32* %10, align 4
  store i32 %11, i32* %4, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @testPointerToArray() #0 {
  %1 = alloca [3 x [3 x i32]], align 16
  %2 = alloca [3 x i32]*, align 8
  %3 = alloca i32, align 4
  %4 = bitcast [3 x [3 x i32]]* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %4, i8* align 16 bitcast ([3 x [3 x i32]]* @__const.testPointerToArray.matrix to i8*), i64 36, i1 false)
  %5 = getelementptr inbounds [3 x [3 x i32]], [3 x [3 x i32]]* %1, i64 0, i64 0
  store [3 x i32]* %5, [3 x i32]** %2, align 8
  %6 = load [3 x i32]*, [3 x i32]** %2, align 8
  %7 = getelementptr inbounds [3 x i32], [3 x i32]* %6, i64 1
  %8 = getelementptr inbounds [3 x i32], [3 x i32]* %7, i64 0, i64 0
  %9 = getelementptr inbounds i32, i32* %8, i64 2
  %10 = load i32, i32* %9, align 4
  store i32 %10, i32* %3, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @testPointerDifference() #0 {
  %1 = alloca [4 x i32], align 16
  %2 = alloca i32*, align 8
  %3 = alloca i32*, align 8
  %4 = alloca i32, align 4
  %5 = bitcast [4 x i32]* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %5, i8* align 16 bitcast ([4 x i32]* @__const.testPointerDifference.numbers to i8*), i64 16, i1 false)
  %6 = getelementptr inbounds [4 x i32], [4 x i32]* %1, i64 0, i64 0
  store i32* %6, i32** %2, align 8
  %7 = getelementptr inbounds [4 x i32], [4 x i32]* %1, i64 0, i64 0
  %8 = getelementptr inbounds i32, i32* %7, i64 2
  store i32* %8, i32** %3, align 8
  %9 = load i32*, i32** %3, align 8
  %10 = load i32*, i32** %2, align 8
  %11 = ptrtoint i32* %9 to i64
  %12 = ptrtoint i32* %10 to i64
  %13 = sub i64 %11, %12
  %14 = sdiv exact i64 %13, 4
  %15 = trunc i64 %14 to i32
  store i32 %15, i32* %4, align 4
  ret void
}

; Function Attrs: noinline nounwind optnone uwtable
define dso_local void @testPointerToArrayElement() #0 {
  %1 = alloca [2 x [2 x i32]], align 16
  %2 = alloca i32*, align 8
  %3 = alloca i32, align 4
  %4 = bitcast [2 x [2 x i32]]* %1 to i8*
  call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 16 %4, i8* align 16 bitcast ([2 x [2 x i32]]* @__const.testPointerToArrayElement.grid to i8*), i64 16, i1 false)
  %5 = getelementptr inbounds [2 x [2 x i32]], [2 x [2 x i32]]* %1, i64 0, i64 1
  %6 = getelementptr inbounds [2 x i32], [2 x i32]* %5, i64 0, i64 0
  store i32* %6, i32** %2, align 8
  %7 = load i32*, i32** %2, align 8
  %8 = getelementptr inbounds i32, i32* %7, i64 1
  %9 = load i32, i32* %8, align 4
  store i32 %9, i32* %3, align 4
  ret void
}

attributes #0 = { noinline nounwind optnone uwtable "frame-pointer"="all" "min-legal-vector-width"="0" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }
attributes #1 = { argmemonly nofree nounwind willreturn }
attributes #2 = { "frame-pointer"="all" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="x86-64" "target-features"="+cx8,+fxsr,+mmx,+sse,+sse2,+x87" "tune-cpu"="generic" }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 1, !"wchar_size", i32 4}
!1 = !{i32 7, !"PIC Level", i32 2}
!2 = !{i32 7, !"PIE Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 2}
!5 = !{!"Ubuntu clang version 14.0.0-1ubuntu1.1"}
!6 = distinct !{!6, !7}
!7 = !{!"llvm.loop.mustprogress"}
!8 = distinct !{!8, !7}
!9 = distinct !{!9, !7}
!10 = distinct !{!10, !7}
!11 = distinct !{!11, !7}
!12 = distinct !{!12, !7}
!13 = distinct !{!13, !7}
