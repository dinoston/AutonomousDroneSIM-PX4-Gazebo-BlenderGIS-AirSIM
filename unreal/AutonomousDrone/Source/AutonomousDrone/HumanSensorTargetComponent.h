// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/EngineTypes.h"
#include "HumanSensorTargetComponent.generated.h"

/**
 * Adds LiDAR/Radar-style recognition debug boxes to a Character Blueprint.
 * Character Blueprint에 LiDAR/Radar 방식의 인식 디버그 박스를 추가합니다.
 */
UCLASS(ClassGroup = (AutonomousDrone), meta = (BlueprintSpawnableComponent))
class AUTONOMOUSDRONE_API UHumanSensorTargetComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UHumanSensorTargetComponent();

	virtual void TickComponent(
		float DeltaTime,
		ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

	/** Enable range and line-of-sight recognition. / 거리 및 시야 인식을 활성화합니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Human Detection")
	bool bEnableDetection = true;

	/** Draw sensor boxes in the game viewport. / 게임 화면에 센서 박스를 표시합니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Human Detection")
	bool bDrawDetectionDebug = true;

	/** LiDAR recognition range in centimetres. / LiDAR 사람 인식 거리(cm)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Human Detection", meta = (ClampMin = "100.0"))
	float LidarDetectionRangeCm = 5000.0f;

	/** Radar recognition range in centimetres. / Radar 사람 인식 거리(cm)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Human Detection", meta = (ClampMin = "100.0"))
	float RadarDetectionRangeCm = 20000.0f;

	/** Detection refresh period in seconds. / 탐지 갱신 주기(초)입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Human Detection", meta = (ClampMin = "0.02"))
	float DetectionUpdateIntervalSeconds = 0.1f;

	/** Collision channel used for line of sight. / 시야 레이가 사용하는 충돌 채널입니다. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Human Detection")
	TEnumAsByte<ECollisionChannel> DetectionTraceChannel = ECC_Visibility;

	/** True while the drone has direct line of sight. / 드론과 시야가 연결되어 있으면 참입니다. */
	UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Human Detection")
	bool bHumanDetected = false;

	/** Latest drone-to-human distance in centimetres. / 최근 드론-사람 거리(cm)입니다. */
	UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Human Detection")
	float DetectedDistanceCm = -1.0f;

protected:
	virtual void BeginPlay() override;

private:
	class APawn* FindObserverDrone() const;
	void UpdateDetection(float DeltaTime);
	void DrawDetectedHuman(float DurationSeconds, bool bDrawLidarBox, bool bDrawRadarBox) const;

	float DetectionUpdateAccumulator = 0.0f;
};
