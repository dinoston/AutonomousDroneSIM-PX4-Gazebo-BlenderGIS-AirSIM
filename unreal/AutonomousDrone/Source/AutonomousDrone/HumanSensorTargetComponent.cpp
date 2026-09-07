// Copyright Epic Games, Inc. All Rights Reserved.

#include "HumanSensorTargetComponent.h"

#include "DrawDebugHelpers.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "FlyingNPCPawn.h"
#include "Components/CapsuleComponent.h"
#include "GameFramework/Character.h"
#include "GameFramework/Pawn.h"
#include "HAL/IConsoleManager.h"
#include "Kismet/GameplayStatics.h"

namespace
{
	bool IsDebugSensorEnabled(const TCHAR* VariableName)
	{
		const IConsoleVariable* Variable = IConsoleManager::Get().FindConsoleVariable(VariableName);
		return Variable == nullptr || Variable->GetInt() != 0;
	}

	bool GetHumanCapsuleBounds(const AActor* Owner, FVector& OutCenter, FVector& OutExtent)
	{
		const ACharacter* Character = Cast<ACharacter>(Owner);
		const UCapsuleComponent* Capsule = Character != nullptr
			? Character->GetCapsuleComponent()
			: nullptr;
		if (Capsule == nullptr)
		{
			return false;
		}

		const float Radius = Capsule->GetScaledCapsuleRadius();
		const float HalfHeight = Capsule->GetScaledCapsuleHalfHeight();
		if (Radius <= KINDA_SMALL_NUMBER || HalfHeight <= KINDA_SMALL_NUMBER)
		{
			return false;
		}

		OutCenter = Capsule->GetComponentLocation();
		OutExtent = FVector(Radius, Radius, HalfHeight);
		return true;
	}
}

UHumanSensorTargetComponent::UHumanSensorTargetComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void UHumanSensorTargetComponent::BeginPlay()
{
	Super::BeginPlay();

	if (AActor* Owner = GetOwner())
	{
		// The tag is also used by Python/AirSim ground-truth filtering.
		// 이 태그는 Python/AirSim 정답 객체 필터에서도 사용합니다.
		Owner->Tags.AddUnique(TEXT("HumanTarget"));
	}
}

void UHumanSensorTargetComponent::TickComponent(
	float DeltaTime,
	ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	UpdateDetection(DeltaTime);
}

void UHumanSensorTargetComponent::UpdateDetection(float DeltaTime)
{
	if (!bEnableDetection)
	{
		bHumanDetected = false;
		DetectedDistanceCm = -1.0f;
		return;
	}

	DetectionUpdateAccumulator += DeltaTime;
	const float UpdateInterval = FMath::Max(0.02f, DetectionUpdateIntervalSeconds);
	if (DetectionUpdateAccumulator < UpdateInterval)
	{
		return;
	}
	DetectionUpdateAccumulator = 0.0f;

	AActor* Owner = GetOwner();
	APawn* ObserverDrone = FindObserverDrone();
	UWorld* World = GetWorld();
	if (Owner == nullptr || ObserverDrone == nullptr || World == nullptr)
	{
		bHumanDetected = false;
		DetectedDistanceCm = -1.0f;
		return;
	}

	FVector HumanCenter;
	FVector HumanExtent;
	if (!GetHumanCapsuleBounds(Owner, HumanCenter, HumanExtent))
	{
		bHumanDetected = false;
		DetectedDistanceCm = -1.0f;
		return;
	}

	const FVector RayStart = ObserverDrone->GetActorLocation();
	const FVector RayEnd = HumanCenter;
	DetectedDistanceCm = FVector::Distance(RayStart, RayEnd);
	const bool bLidarEnabled = IsDebugSensorEnabled(TEXT("autodrone.LidarDebug"));
	const bool bRadarEnabled = IsDebugSensorEnabled(TEXT("autodrone.RadarDebug"));
	const bool bWithinLidarRange = DetectedDistanceCm <= LidarDetectionRangeCm;
	const bool bWithinRadarRange = DetectedDistanceCm <= RadarDetectionRangeCm;
	if ((!bLidarEnabled || !bWithinLidarRange) && (!bRadarEnabled || !bWithinRadarRange))
	{
		bHumanDetected = false;
		return;
	}

	FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(HumanSensorDetectionRay), true, ObserverDrone);
	QueryParams.AddIgnoredActor(ObserverDrone);
	FHitResult Hit;
	const bool bHit = World->LineTraceSingleByChannel(
		Hit,
		RayStart,
		RayEnd,
		DetectionTraceChannel,
		QueryParams);
	bHumanDetected = !bHit || Hit.GetActor() == Owner;

	const bool bDrawLidarBox = bLidarEnabled && bWithinLidarRange;
	const bool bDrawRadarBox = bRadarEnabled && bWithinRadarRange;
	if (!bDrawDetectionDebug || (!bDrawLidarBox && !bDrawRadarBox))
	{
		return;
	}

	const float DrawDuration = UpdateInterval * 1.25f;
	const FVector VisibleRayEnd = bHit ? Hit.ImpactPoint : RayEnd;
	const FColor RayColor = !bHumanDetected
		? FColor::Yellow
		: (bDrawLidarBox ? FColor::Green : FColor(36, 148, 255));
	if (IsDebugSensorEnabled(TEXT("autodrone.SensorRayDebug")))
	{
		DrawDebugLine(World, RayStart, VisibleRayEnd, RayColor, false, DrawDuration, 0, 2.0f);
	}
	if (bHumanDetected)
	{
		DrawDetectedHuman(DrawDuration, bDrawLidarBox, bDrawRadarBox);
	}
}

APawn* UHumanSensorTargetComponent::FindObserverDrone() const
{
	AActor* Owner = GetOwner();
	APawn* PlayerPawn = UGameplayStatics::GetPlayerPawn(this, 0);
	if (PlayerPawn != nullptr
		&& PlayerPawn != Owner
		&& !PlayerPawn->IsA<ACharacter>()
		&& !PlayerPawn->IsA<AFlyingNPCPawn>())
	{
		return PlayerPawn;
	}

	// AirSim may control a pawn without normal player possession. Ignore all
	// Characters so one pedestrian is never selected as another's sensor.
	// AirSim 기체는 일반 PlayerController가 점유하지 않을 수 있습니다. 한 사람이
	// 다른 사람의 센서가 되지 않도록 모든 Character를 관측 후보에서 제외합니다.
	APawn* ClosestDrone = nullptr;
	float ClosestDistanceSquared = TNumericLimits<float>::Max();
	for (TActorIterator<APawn> It(GetWorld()); It; ++It)
	{
		APawn* Candidate = *It;
		if (Candidate == nullptr
			|| Candidate == Owner
			|| Candidate->IsA<ACharacter>()
			|| Candidate->IsA<AFlyingNPCPawn>())
		{
			continue;
		}
		const float DistanceSquared = FVector::DistSquared(
			Candidate->GetActorLocation(),
			Owner->GetActorLocation());
		if (DistanceSquared < ClosestDistanceSquared)
		{
			ClosestDistanceSquared = DistanceSquared;
			ClosestDrone = Candidate;
		}
	}
	return ClosestDrone;
}

void UHumanSensorTargetComponent::DrawDetectedHuman(
	float DurationSeconds,
	bool bDrawLidarBox,
	bool bDrawRadarBox) const
{
	const AActor* Owner = GetOwner();
	UWorld* World = GetWorld();
	if (Owner == nullptr || World == nullptr)
	{
		return;
	}

	FVector Center;
	FVector Extent;
	if (!GetHumanCapsuleBounds(Owner, Center, Extent))
	{
		return;
	}
	if (bDrawLidarBox)
	{
		DrawDebugBox(
			World,
			Center,
			Extent,
			FQuat::Identity,
			FColor::Green,
			false,
			DurationSeconds,
			0,
			1.5f);
	}
	if (bDrawRadarBox)
	{
		// Radar stays blue and only slightly larger than the character capsule.
		// Radar 박스는 파란색이며 캐릭터 캡슐보다 조금만 크게 표시합니다.
		DrawDebugBox(
			World,
			Center,
			Extent * 1.08f + FVector(2.0f),
			FQuat::Identity,
			FColor(36, 148, 255),
			false,
			DurationSeconds,
			0,
			1.5f);
	}

	const FColor LabelColor = bDrawLidarBox ? FColor::Green : FColor(36, 148, 255);
	const TCHAR* SensorLabel = bDrawLidarBox && bDrawRadarBox
		? TEXT("LIDAR + RADAR")
		: (bDrawRadarBox ? TEXT("RADAR") : TEXT("LIDAR"));
	DrawDebugString(
		World,
		Center + FVector(0.0f, 0.0f, Extent.Z + 35.0f),
		FString::Printf(TEXT("%s  HUMAN  %.1f m"), SensorLabel, DetectedDistanceCm / 100.0f),
		nullptr,
		LabelColor,
		DurationSeconds,
		true,
		1.1f);
}
