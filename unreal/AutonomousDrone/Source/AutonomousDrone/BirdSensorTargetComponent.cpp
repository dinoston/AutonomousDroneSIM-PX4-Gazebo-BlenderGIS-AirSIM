// Copyright Epic Games, Inc. All Rights Reserved.

#include "BirdSensorTargetComponent.h"

#include "Components/SkeletalMeshComponent.h"
#include "Components/StaticMeshComponent.h"
#include "DrawDebugHelpers.h"
#include "EngineUtils.h"
#include "Engine/World.h"
#include "FlyingNPCPawn.h"
#include "GameFramework/Character.h"
#include "GameFramework/Pawn.h"
#include "HAL/IConsoleManager.h"
#include "Kismet/GameplayStatics.h"

namespace
{
	bool IsBirdDebugSensorEnabled(const TCHAR* VariableName)
	{
		const IConsoleVariable* Variable = IConsoleManager::Get().FindConsoleVariable(VariableName);
		return Variable == nullptr || Variable->GetInt() != 0;
	}
}

UBirdSensorTargetComponent::UBirdSensorTargetComponent()
{
	PrimaryComponentTick.bCanEverTick = true;
}

void UBirdSensorTargetComponent::BeginPlay()
{
	Super::BeginPlay();
	if (AActor* Owner = GetOwner())
	{
		Owner->Tags.AddUnique(TEXT("BirdTarget"));
	}
}

void UBirdSensorTargetComponent::TickComponent(
	float DeltaTime,
	ELevelTick TickType,
	FActorComponentTickFunction* ThisTickFunction)
{
	Super::TickComponent(DeltaTime, TickType, ThisTickFunction);
	UpdateDetection(DeltaTime);
}

bool UBirdSensorTargetComponent::GetBirdMeshBounds(
	FVector& OutCenter,
	FVector& OutExtent) const
{
	AActor* Owner = GetOwner();
	if (Owner == nullptr)
	{
		return false;
	}

	TInlineComponentArray<USkeletalMeshComponent*> SkeletalMeshes;
	Owner->GetComponents(SkeletalMeshes);
	for (const USkeletalMeshComponent* Mesh : SkeletalMeshes)
	{
		if (Mesh != nullptr && Mesh->GetSkeletalMeshAsset() != nullptr)
		{
			OutCenter = Mesh->Bounds.Origin;
			OutExtent = Mesh->Bounds.BoxExtent.ComponentMax(FVector(2.0f));
			return true;
		}
	}

	TInlineComponentArray<UStaticMeshComponent*> StaticMeshes;
	Owner->GetComponents(StaticMeshes);
	for (const UStaticMeshComponent* Mesh : StaticMeshes)
	{
		if (Mesh != nullptr && Mesh->GetStaticMesh() != nullptr)
		{
			OutCenter = Mesh->Bounds.Origin;
			OutExtent = Mesh->Bounds.BoxExtent.ComponentMax(FVector(2.0f));
			return true;
		}
	}

	const FBox Bounds = Owner->GetComponentsBoundingBox(true);
	if (!Bounds.IsValid)
	{
		return false;
	}
	OutCenter = Bounds.GetCenter();
	OutExtent = Bounds.GetExtent().ComponentMax(FVector(2.0f));
	return true;
}

void UBirdSensorTargetComponent::UpdateDetection(float DeltaTime)
{
	if (!bEnableDetection)
	{
		bBirdDetected = false;
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
	FVector BirdCenter;
	FVector BirdExtent;
	if (Owner == nullptr || ObserverDrone == nullptr || World == nullptr
		|| !GetBirdMeshBounds(BirdCenter, BirdExtent))
	{
		bBirdDetected = false;
		DetectedDistanceCm = -1.0f;
		return;
	}

	const FVector RayStart = ObserverDrone->GetActorLocation();
	DetectedDistanceCm = FVector::Distance(RayStart, BirdCenter);
	const bool bLidarEnabled = IsBirdDebugSensorEnabled(TEXT("autodrone.LidarDebug"));
	const bool bRadarEnabled = IsBirdDebugSensorEnabled(TEXT("autodrone.RadarDebug"));
	const bool bWithinLidarRange = DetectedDistanceCm <= LidarDetectionRangeCm;
	const bool bWithinRadarRange = DetectedDistanceCm <= RadarDetectionRangeCm;
	if ((!bLidarEnabled || !bWithinLidarRange) && (!bRadarEnabled || !bWithinRadarRange))
	{
		bBirdDetected = false;
		return;
	}

	FCollisionQueryParams QueryParams(SCENE_QUERY_STAT(BirdSensorDetectionRay), true, ObserverDrone);
	QueryParams.AddIgnoredActor(ObserverDrone);
	FHitResult Hit;
	const bool bHit = World->LineTraceSingleByChannel(
		Hit,
		RayStart,
		BirdCenter,
		DetectionTraceChannel,
		QueryParams);
	bBirdDetected = !bHit || Hit.GetActor() == Owner;

	const bool bDrawLidarBox = bLidarEnabled && bWithinLidarRange;
	const bool bDrawRadarBox = bRadarEnabled && bWithinRadarRange;
	if (!bDrawDetectionDebug || (!bDrawLidarBox && !bDrawRadarBox))
	{
		return;
	}

	const float DrawDuration = UpdateInterval * 1.25f;
	const FVector VisibleRayEnd = bHit ? Hit.ImpactPoint : BirdCenter;
	const FColor RayColor = !bBirdDetected
		? FColor::Yellow
		: (bDrawLidarBox ? FColor::Green : FColor(36, 148, 255));
	if (IsBirdDebugSensorEnabled(TEXT("autodrone.SensorRayDebug")))
	{
		DrawDebugLine(World, RayStart, VisibleRayEnd, RayColor, false, DrawDuration, 0, 1.0f);
	}
	if (bBirdDetected)
	{
		DrawDetectedBird(DrawDuration, bDrawLidarBox, bDrawRadarBox);
	}
}

APawn* UBirdSensorTargetComponent::FindObserverDrone() const
{
	AActor* Owner = GetOwner();
	UWorld* World = GetWorld();
	if (World == nullptr)
	{
		return nullptr;
	}
	APawn* PlayerPawn = UGameplayStatics::GetPlayerPawn(this, 0);
	if (PlayerPawn != nullptr
		&& PlayerPawn != Owner
		&& !PlayerPawn->IsA<ACharacter>()
		&& !PlayerPawn->IsA<AFlyingNPCPawn>())
	{
		return PlayerPawn;
	}

	APawn* ClosestDrone = nullptr;
	float ClosestDistanceSquared = TNumericLimits<float>::Max();
	for (TActorIterator<APawn> It(World); It; ++It)
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

void UBirdSensorTargetComponent::DrawDetectedBird(
	float DurationSeconds,
	bool bDrawLidarBox,
	bool bDrawRadarBox) const
{
	UWorld* World = GetWorld();
	FVector Center;
	FVector Extent;
	if (World == nullptr || !GetBirdMeshBounds(Center, Extent))
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
			1.0f);
	}
	if (bDrawRadarBox)
	{
		DrawDebugBox(
			World,
			Center,
			Extent * 1.15f + FVector(2.0f),
			FQuat::Identity,
			FColor(36, 148, 255),
			false,
			DurationSeconds,
			0,
			2.0f);
	}

	const FColor LabelColor = bDrawLidarBox ? FColor::Green : FColor(36, 148, 255);
	const TCHAR* SensorLabel = bDrawLidarBox && bDrawRadarBox
		? TEXT("LIDAR + RADAR")
		: (bDrawRadarBox ? TEXT("RADAR") : TEXT("LIDAR"));
	DrawDebugString(
		World,
		Center + FVector(0.0f, 0.0f, Extent.Z + 12.0f),
		FString::Printf(TEXT("%s  BIRD  %.1f m"), SensorLabel, DetectedDistanceCm / 100.0f),
		nullptr,
		LabelColor,
		DurationSeconds,
		true,
		0.8f);
}
