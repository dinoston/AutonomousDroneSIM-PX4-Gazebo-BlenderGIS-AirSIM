// Copyright Epic Games, Inc. All Rights Reserved.

#pragma once

#include "CoreMinimal.h"
#include "Components/ActorComponent.h"
#include "Engine/EngineTypes.h"
#include "BirdSensorTargetComponent.generated.h"

/**
 * Adds LiDAR/Radar recognition debug boxes to flocking bird actors.
 * 군집 새 액터에 LiDAR/Radar 인식 디버그 박스를 추가합니다.
 */
UCLASS(ClassGroup = (AutonomousDrone), meta = (BlueprintSpawnableComponent))
class AUTONOMOUSDRONE_API UBirdSensorTargetComponent : public UActorComponent
{
	GENERATED_BODY()

public:
	UBirdSensorTargetComponent();

	virtual void TickComponent(
		float DeltaTime,
		ELevelTick TickType,
		FActorComponentTickFunction* ThisTickFunction) override;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bird Detection")
	bool bEnableDetection = true;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bird Detection")
	bool bDrawDetectionDebug = true;

	/** LiDAR range: 80 m. / LiDAR 새 인식 거리: 80m. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bird Detection", meta = (ClampMin = "100.0"))
	float LidarDetectionRangeCm = 8000.0f;

	/** Long-range Radar range: 1 km. / 장거리 Radar 새 인식 거리: 1km. */
	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bird Detection", meta = (ClampMin = "100.0"))
	float RadarDetectionRangeCm = 100000.0f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bird Detection", meta = (ClampMin = "0.02"))
	float DetectionUpdateIntervalSeconds = 0.1f;

	UPROPERTY(EditAnywhere, BlueprintReadWrite, Category = "Bird Detection")
	TEnumAsByte<ECollisionChannel> DetectionTraceChannel = ECC_Visibility;

	UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Bird Detection")
	bool bBirdDetected = false;

	UPROPERTY(VisibleInstanceOnly, BlueprintReadOnly, Category = "Bird Detection")
	float DetectedDistanceCm = -1.0f;

protected:
	virtual void BeginPlay() override;

private:
	class APawn* FindObserverDrone() const;
	bool GetBirdMeshBounds(FVector& OutCenter, FVector& OutExtent) const;
	void UpdateDetection(float DeltaTime);
	void DrawDetectedBird(float DurationSeconds, bool bDrawLidarBox, bool bDrawRadarBox) const;

	float DetectionUpdateAccumulator = 0.0f;
};
