// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Engine/EngineTypes.h"
#include "GameFramework/Pawn.h"
#include "FlyingNPCPawn.generated.h"

class AActor;
class APawn;
class UPrimitiveComponent;
class USphereComponent;
class UStaticMeshComponent;
class UFloatingPawnMovement;
struct FHitResult;

/**
 * Reusable flying NPC pawn for birds and drones.
 * 새와 드론에 공통으로 사용할 수 있는 비행 NPC Pawn입니다.
 */
UCLASS(Blueprintable)
class AUTONOMOUSDRONE_API AFlyingNPCPawn : public APawn
{
	GENERATED_BODY()

public:
	AFlyingNPCPawn();

	virtual void Tick(float DeltaSeconds) override;
	virtual UPawnMovementComponent* GetMovementComponent() const override;

	/** Start or resume waypoint patrol. / 웨이포인트 순찰을 시작하거나 재개합니다. */
	UFUNCTION(BlueprintCallable, Category = "Flying NPC|Patrol")
	void StartPatrol();

	/** Stop movement without deleting the route. / 경로를 유지한 채 이동을 정지합니다. */
	UFUNCTION(BlueprintCallable, Category = "Flying NPC|Patrol")
	void StopPatrol();

	/** Temporarily fly to a world location. / 월드 좌표를 임시 목적지로 지정합니다. */
	UFUNCTION(BlueprintCallable, Category = "Flying NPC|Patrol")
	void SetDirectTarget(const FVector& WorldLocation);

	/** Clear the temporary target and resume patrol. / 임시 목적지를 지우고 순찰로 복귀합니다. */
	UFUNCTION(BlueprintCallable, Category = "Flying NPC|Patrol")
	void ClearDirectTarget();

	/** Collision root used by movement and avoidance. / 이동과 회피에 사용하는 충돌 루트입니다. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Flying NPC|Components")
	TObjectPtr<USphereComponent> CollisionSphere;

	/** Assign the imported FBX mesh in a Blueprint child. / Blueprint 자식에서 가져온 FBX 메시를 지정합니다. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Flying NPC|Components")
	TObjectPtr<UStaticMeshComponent> DroneMesh;

	/** Built-in acceleration-based pawn movement. / 가속 기반 기본 Pawn 이동 컴포넌트입니다. */
	UPROPERTY(VisibleAnywhere, BlueprintReadOnly, Category = "Flying NPC|Components")
	TObjectPtr<UFloatingPawnMovement> FlightMovement;

	/** Ordered 3D patrol points placed in the level. / 레벨에 배치한 순서형 3D 순찰 지점입니다. */
	UPROPERTY(EditInstanceOnly, BlueprintReadWrite, Category = "Flying NPC|Patrol")
	TArray<TObjectPtr<AActor>> PatrolPoints;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Patrol")
	bool bStartPatrolOnBeginPlay = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Patrol")
	bool bLoopPatrol = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Patrol")
	bool bRandomizePatrol = false;

	/** Target acceptance radius in centimetres. / 목적지 도착 판정 반경(cm)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Patrol", meta = (ClampMin = "10.0"))
	float AcceptanceRadiusCm = 150.0f;

	/** Maximum flight speed in centimetres per second. / 최대 비행 속도(cm/s)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Movement", meta = (ClampMin = "10.0"))
	float MaxFlightSpeedCmPerSecond = 500.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Movement", meta = (ClampMin = "10.0"))
	float AccelerationCmPerSecondSquared = 700.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Movement", meta = (ClampMin = "10.0"))
	float DecelerationCmPerSecondSquared = 900.0f;

	/** Smooth yaw rotation speed. / 부드러운 Yaw 회전 보간 속도입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Movement", meta = (ClampMin = "0.1"))
	float RotationInterpSpeed = 3.0f;

	/** Obstacle probe length in centimetres. / 장애물 탐지 거리(cm)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Avoidance", meta = (ClampMin = "100.0"))
	float AvoidanceTraceDistanceCm = 1200.0f;

	/** Radius of each swept-sphere probe. / 장애물 탐지 구체의 반경(cm)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Avoidance", meta = (ClampMin = "5.0"))
	float AvoidanceProbeRadiusCm = 100.0f;

	/** Weight of the selected escape direction. / 선택한 회피 방향에 적용할 가중치입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Avoidance", meta = (ClampMin = "0.1"))
	float AvoidanceStrength = 2.2f;

	/** Keep one escape choice briefly to prevent left/right jitter. / 좌우 떨림 방지를 위해 회피 방향을 유지하는 시간입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Avoidance", meta = (ClampMin = "0.0"))
	float AvoidanceCommitSeconds = 0.8f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Avoidance")
	TEnumAsByte<ECollisionChannel> ObstacleTraceChannel = ECC_Visibility;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Debug")
	bool bDrawAvoidanceDebug = false;

	/** Mark this pawn as an enemy target at runtime. / 실행 중 이 Pawn을 적 표적으로 표시합니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Detection")
	bool bIsEnemyTarget = true;

	/** Enable player-to-enemy visibility raycasts. / 플레이어에서 적까지의 시야 레이캐스트를 활성화합니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Detection")
	bool bEnableProximityDetection = true;

	/** Maximum recognition distance in centimetres. / 적 인식 최대 거리(cm)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Detection", meta = (ClampMin = "100.0"))
	float DetectionRangeCm = 5000.0f;

	/** Longer Radar visualization range in centimetres. / Radar 시각화의 장거리 범위(cm)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Detection", meta = (ClampMin = "100.0"))
	float RadarDetectionRangeCm = 20000.0f;

	/** Detection refresh period. / 적 탐지 갱신 주기(초)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Detection", meta = (ClampMin = "0.02"))
	float DetectionUpdateIntervalSeconds = 0.1f;

	/** Collision channel used by the line-of-sight ray. / 시야 레이가 사용하는 충돌 채널입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Detection")
	TEnumAsByte<ECollisionChannel> DetectionTraceChannel = ECC_Visibility;

	/** Draw the recognition ray, 3D box, and range in the game viewport. / 게임 화면에 인식 레이, 3D 박스와 거리를 표시합니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Flying NPC|Detection")
	bool bDrawDetectionDebug = true;

	/** True while the player drone has direct line of sight. / 플레이어 드론과 시야가 연결되어 있으면 참입니다. */
	UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Flying NPC|Detection")
	bool bTargetDetected = false;

	/** Latest player-to-target distance in centimetres. / 최근 플레이어-표적 거리(cm)입니다. */
	UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Flying NPC|Detection")
	float DetectedDistanceCm = -1.0f;

protected:
	virtual void BeginPlay() override;

	UFUNCTION()
	void HandleCollisionHit(
		UPrimitiveComponent* HitComponent,
		AActor* OtherActor,
		UPrimitiveComponent* OtherComponent,
		FVector NormalImpulse,
		const FHitResult& Hit);

private:
	bool ResolveTargetLocation(FVector& OutTargetLocation) const;
	void AdvancePatrolPoint();
	FVector ComputeSteeringDirection(const FVector& DesiredDirection, float DeltaSeconds);
	float TraceClearance(const FVector& Direction, float DistanceCm, bool bDebug) const;
	void ApplyMovementSettings();
	void UpdateProximityDetection(float DeltaSeconds);
	APawn* FindObserverPawn() const;
	void DrawDetectedTarget(float DurationSeconds, bool bDrawLidarBox, bool bDrawRadarBox) const;

	int32 PatrolPointIndex = 0;
	bool bPatrolActive = false;
	bool bHasDirectTarget = false;
	FVector DirectTargetLocation = FVector::ZeroVector;
	FVector CommittedAvoidanceDirection = FVector::ZeroVector;
	float AvoidanceCommitRemaining = 0.0f;
	float DetectionUpdateAccumulator = 0.0f;
};
